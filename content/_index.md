---
# Leave the homepage title empty to use the site title
title: ''
summary: ''
date: 2022-10-24
type: landing

design:
  # Default section spacing
  spacing: '6rem'

sections:
  - block: resume-biography-3
    content:
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: me
      text: |-
        Hi, I'm Louis. I'm a research scientist at Meta on the [Adaptive
        Experimentation](/tags/adaptive-experimentation/) team within Central
        Applied Science (CAS), based in New York City. My research is in
        probabilistic machine learning — approximate Bayesian inference,
        [Gaussian processes](/tags/gaussian-processes/), and [Bayesian
        optimization](/tags/bayesian-optimization/). I obtained my PhD at
        the [University of Sydney](https://www.sydney.edu.au/), advised
        by [Edwin Bonilla](#) and [Fabio Ramos](#).
      # Show a call-to-action button under your biography? (optional)
      button:
        text: Curriculum Vitae (CV)
        url: uploads/cv-louis-tiao.pdf
      headings:
        about: 'About'
        education: ''
        interests: ''
    design:
      # Use the new Gradient Mesh which automatically adapts to the selected theme colors
      background:
        gradient_mesh:
          enable: true

      # Name heading sizing to accommodate long or short names
      name:
        size: sm # Options: xs, sm, md, lg (default), xl

      # Avatar customization
      avatar:
        size: medium # Options: small (150px), medium (200px, default), large (320px), xl (400px), xxl (500px)
        shape: circle # Options: circle (default), square, rounded
  - block: markdown
    content:
      title: 'My Research'
      subtitle: ''
      text: |-
        I work on probabilistic machine learning, with particular focus on
        approximate Bayesian inference and [Gaussian
        processes](/tags/gaussian-processes/), and their applications to
        [Bayesian optimization](/tags/bayesian-optimization/). More
        broadly, my interests extend to [automated machine learning
        (AutoML)](/tags/automl/), encompassing [hyperparameter
        optimization](/tags/hyperparameter-optimization/) and adaptive
        resource allocation techniques such as early stopping and
        scaling laws. Past work includes [graph representation
        learning](/tags/graph-representation-learning/),
        [GANs](/tags/generative-adversarial-networks/), and [implicit
        generative models](/tags/generative-models/).
    design:
      columns: '1'
  - block: collection
    id: news
    content:
      title: News
      subtitle: ''
      text: ''
      # Choose how many pages you would like to display (0 = all pages)
      count: 5
      # Page type to display. E.g. post, talk, publication...
      page_type: posts
      # Filter on criteria
      filters:
        author: ''
        category: 'news'
        tag: ''
        exclude_featured: false
        exclude_future: false
        exclude_past: false
      # Choose how many pages you would like to offset by
      offset: 0
      # Page order: descending (desc) or ascending (asc) date.
      order: desc
    design:
      # Choose a layout view
      view: date-title-summary
  - block: collection
    id: publications
    content:
      title: Featured Publications
      count: 6
      filters:
        folders:
          - publications
        featured_only: true
    design:
      view: article-grid
      columns: 2
  - block: collection
    id: posts
    content:
      title: Recent Posts
      subtitle: ''
      text: ''
      # Choose how many pages you would like to display (0 = all pages)
      count: 8
      # Page type to display. E.g. post, talk, publication...
      page_type: posts
      # Filter on criteria
      filters:
        author: ''
        category: 'technical'
        tag: ''
        exclude_featured: false
        exclude_future: false
        exclude_past: false
      # Choose how many pages you would like to offset by
      offset: 0
      # Page order: descending (desc) or ascending (asc) date.
      order: desc
    design:
      # Choose a layout view
      view: article-grid
      columns: 2
  - block: collection
    id: talks
    content:
      title: Recent & Upcoming Talks
      filters:
        folders:
          - events
    design:
      view: card
  - block: collection
    id: projects
    content:
      title: Featured Projects
      count: 3
      filters:
        folders:
          - projects
        featured_only: true
    design:
      view: article-grid
      columns: 3
      fill_image: false
      show_date: false
      show_read_time: false
      show_read_more: false
  - block: contact-info
    id: contact
    content:
      title: Contact
      subtitle: ''
      text: Drop me a line
      email: louis@tiao.io
      autolink: true
---
