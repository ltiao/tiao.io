---
aliases:
  - /post/efficient-cholesky-decomposition-of-low-rank-updates/
title: Efficient Cholesky decomposition of low-rank updates
subtitle: A short and practical guide to efficiently computing the Cholesky
  decomposition of matrices perturbed by low-rank updates
date: 2023-04-16T11:16:03.167Z
summary: A short and practical guide to efficiently computing the Cholesky
  decomposition of matrices perturbed by low-rank updates
draft: false
featured: true
authors:
  - me
math: true
tags:
  - TensorFlow Probability
  - Machine Learning
  - Linear Algebra
  - Gaussian Processes
categories:
  - technical
image:
  filename: paul-minami-tzph9aswxu-unsplash.jpg
  focal_point: Smart
  preview_only: true
---

Suppose we're given a positive semidefinite (PSD) 
matrix {{< math >}}$\mathbf{A} \in \mathbb{R}^{N \times N}${{< /math >}} to
which we wish to update by some low-rank
matrix {{< math >}}$\mathbf{U} \mathbf{U}^\top \in \mathbb{R}^{N \times N}${{< /math >}},
{{< math >}}
$$\mathbf{B} \triangleq \mathbf{A} + \mathbf{U} \mathbf{U}^\top,$$
{{< /math >}} 
where the update factor matrix {{< math >}}$\mathbf{U} \in \mathbb{R}^{N \times M}${{< /math >}}.
To be more precise, the low-rank update is rank-$M$ for some $M \ll N$.

*What is the best way to calculate the Cholesky decomposition of {{< math >}}$\mathbf{B}${{< /math >}}?*

Given no additional information the obvious way is to calculate it directly, 
which incurs a cost of {{< math >}}$\mathcal{O}(N^3)${{< /math >}}.
But suppose we've already calculated the lower-triangular Cholesky factor 
{{< math >}}$\mathbf{L} \in \mathbb{R}^{N \times N}${{< /math >}} 
of {{< math >}}$\mathbf{A}${{< /math >}} (i.e., {{< math >}}$\mathbf{LL}^\top = \mathbf{A}${{< /math >}}).
Then, we can use it to calculate the Cholesky decomposition 
of {{< math >}}$\mathbf{B}${{< /math >}} at a reduced cost 
of {{< math >}}$\mathcal{O}(N^2M)${{< /math >}}.
Here's how.

## Rank-1 Updates

First, let's consider the simpler case involving just *rank-1 updates*
{{< math >}}
$$\mathbf{B} \triangleq \mathbf{A} + \mathbf{u} \mathbf{u}^\top,$$
{{< /math >}} 
where update factor vector {{< math >}}$\mathbf{u} \in \mathbb{R}^{N}${{< /math >}}.
With some clever manipulations[^seeger2004low], the details of which we won't
get into in this post, we can leverage {{< math >}}$\mathbf{L}${{< /math >}} to 
calculate the Cholesky decomposition of {{< math >}}$\mathbf{B}${{< /math >}} 
at a reduced cost of {{< math >}}$\mathcal{O}(N^2)${{< /math >}}.
Such a procedure for rank-1 updates is implemented in the old-school Fortran 
linear algebra software library [LINPACK](https://netlib.org/linpack/) 
(but unfortunately not in its successor [LAPACK](https://netlib.org/lapack/)),
and also in modern libraries like [TensorFlow Probability](https://www.tensorflow.org/probability) (TFP).

In TFP, this is implemented in the function named [tfp.math.cholesky_update](https://www.tensorflow.org/probability/api_docs/python/tfp/math/cholesky_update). 
For example,

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

update_factor_vector  # Tensor; shape [..., N]
a  # Tensor; shape [..., N, N]

update = tf.linalg.matmul(
    update_factor_vector[..., tf.newaxis],
    update_factor_vector[..., tf.newaxis],
    transpose_b=True
)

b = a + update  # Tensor; shape [..., N, N]
a_factor = tf.linalg.cholesky(a)  # O(N^3); suppose this is pre-computed and stored

b_factor = tf.linalg.cholesky(b)  # O(N^3), ignores `a_factor`
b_factor_1 = tfp.math.cholesky_update(a_factor, update_factor_vector)  # O(N^2), uses `a_factor`

np.testing.assert_array_almost_equal(b_factor, b_factor_1)
```

Here `cholesky_update` takes as arguments `chol` with shape `[B1, ..., Bn, N, N]` 
and `u` with shape `[B1, ..., Bn, N]`, and returns a lower triangular Cholesky 
factor of the rank-1 updated matrix `chol @ chol.T + u @ u.T` in {{< math >}}$\mathcal{O}(N^2)${{< /math >}} time.

## Low-Rank Updates

Now let's return to rank-$M$ updates.
First let's write the update factor matrix $\mathbf{U}$ in terms of column 
vectors $\mathbf{u}_m \in \mathbb{R}^{N}$,
{{< math >}}
$$
\mathbf{U} \triangleq
\begin{bmatrix}
\mathbf{u}_1 & \cdots & \mathbf{u}_M
\end{bmatrix}.
$$
{{< /math >}} 

Now we can write the rank-$M$ update matrix as a sum of $M$ rank-1 matrices,
{{< math >}}
$$
\mathbf{U} \mathbf{U}^\top = 
\begin{bmatrix} \mathbf{u}_1 & \cdots & \mathbf{u}_M \end{bmatrix} 
\begin{bmatrix} \mathbf{u}_1^\top \\ \vdots \\ \mathbf{u}_M^\top \end{bmatrix} = 
\sum_{m=1}^{M} \mathbf{u}_m \mathbf{u}_m^\top.
$$
{{< /math >}} 

```python
update_factor_matrix  # Tensor; shape [..., N, M]

# [..., N, 1, M] [..., 1, N, M] -> [..., N, N, M] -> [..., N, N]
update1 = tf.reduce_sum(update_factor_matrix[..., tf.newaxis, :] *
                        update_factor_matrix[..., tf.newaxis, :, :], axis=-1)
# [..., N, M] [..., M, N] -> [..., N, N]
update2 = tf.linalg.matmul(update_factor_matrix,
                           update_factor_matrix, transpose_b=True)

# not exactly equal due to finite precision, but still equal up to high precision
np.testing.assert_array_almost_equal(update1, update2, decimal=14)
```

Thus seen, a low-rank update is nothing more than a repeated application of 
rank-1 updates,
{{< math >}}
$$
\begin{align}
\mathbf{B} & = \mathbf{A} + \mathbf{U} \mathbf{U}^\top \\ & =
\mathbf{A} + \sum_{m=1}^{M} \mathbf{u}_m \mathbf{u}_m^\top \\ & = 
((\mathbf{A} + \mathbf{u}_1 \mathbf{u}_1^\top) + \cdots ) + \mathbf{u}_M \mathbf{u}_M^{\top}.
\end{align}
$$
{{< /math >}} 

Therefore, we can simply leverage the $O(N^2)$ procedure for Cholesky 
decompositions of rank-1 updates and apply it recursively $M$ times to obtain 
a $O(N^2M)$ procedure for rank-$M$ updates.

Hence, we have:

```python
# [..., N, M] [..., M, N] -> [..., N, N]
update = tf.linalg.matmul(update_factor_matrix,
                          update_factor_matrix, transpose_b=True)
b = a + update  # Tensor; shape [..., N, N]

b_factor = tf.linalg.cholesky(b)  # O(N^3), ignores `a_factor`
b_factor_1 = cholesky_update_iterated(a_factor, update_factor_matrix)  # O(N^2M), uses `a_factor`

np.testing.assert_array_almost_equal(b_factor_1, b_factor)
```

where function `cholesky_update_iterated` is implemented as follows:

```python
def cholesky_update_iterated(chol, update_factor_matrix):

    # base case
    if update_factor_matrix.shape[-1] == 0:
        return chol

    prev = cholesky_update_iterated(chol, update_factor_matrix[..., :-1])
    return tfp.math.cholesky_update(prev, update_factor_matrix[..., -1])
```

We can also implement this iteratively.
First we'd use `tf.unstack` to turn the update factor matrix $\mathbf{U}$ 
into a list of update factor vectors $\mathbf{u}_m$:

```python
>>> update_factor_vectors = tf.unstack(update_factor_matrix, axis=-1)
>>> assert isinstance(update_factor_vectors, list)  # `update_factor_vectors` is a list
>>> assert len(update_factor_vectors) == M  # ... the list contains M vectors
>>> assert update_factor_vectors[0].shape == (*Bs, N)  # ... and each vector has shape [B1, ..., Bn, N]
```

Then, we have:

```python
def cholesky_update_iterated(chol, update_factor_matrix):
    new_chol = chol
    for update_factor_vector in tf.unstack(update_factor_matrix, axis=-1):
        new_chol = tfp.math.cholesky_update(new_chol, update_factor_vector)
    return new_chol
```

The astute reader will recognize that this is simply an special case of 
the [itertools.accumulate](https://docs.python.org/3/library/itertools.html#itertools.accumulate) 
or [functools.reduce](https://docs.python.org/3/library/functools.html#functools.reduce)
patterns, where 
the *binary operator* is `tfp.math.cholesky_update`, 
the *iterable* is `tf.unstack(update_factor, axis=-1)` and 
the *initial value* is `chol`.

Therefore, we can also implement it neatly using the one-liner:

```python
from functools import reduce


def cholesky_update_iterated(chol, update_factor_matrix):
    return reduce(tfp.math.cholesky_update, tf.unstack(update_factor_matrix, axis=-1), chol)
```

## Summary

In summary, we showed that to efficiently calculate the Cholesky decomposition 
of a matrix perturbed by a low-rank update, one just needs to iteratively 
calculate that of the same matrix perturbed by a series of rank-1 updates.
Better yet, all of this can be done with a simple one-liner!

To receive updates on more posts like this, follow me on [Twitter] and [GitHub]!

[Twitter]: https://twitter.com/louistiao
[GitHub]: https://github.com/ltiao


[^seeger2004low]: Seeger, M. (2004). Low rank updates for the Cholesky decomposition.
[^dongarra1979linpack]: Dongarra, J. J., Moler, C. B., Bunch, J. R., & Stewart, G. W. (1979). LINPACK users' guide. Society for Industrial and Applied Mathematics.
