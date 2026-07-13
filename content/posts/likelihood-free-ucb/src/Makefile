# Figures for "UCB in Disguise" — XeLaTeX (Source Serif 4) + pdftoppm at 350 dpi.
#
# The committed *_data.tex files are inputs to cdf-readouts.tex and
# two-sigmas.tex; regenerate them with the gen_*.py scripts:
#   uv run --with numpy python3 gen_cdf_data.py          # writes cdf_data.tex
#   uv run --with numpy python3 gen_two_sigmas_data.py > two_sigmas_data.tex
#   uv run --with numpy python3 gen_ts_collapse_data.py  # prints pgfplots blocks
#                                                        # embedded in ts-collapse.tex
FIGURES = pinball pinball-tilt recipe-schematic ts-collapse level-sets cdf-readouts two-sigmas \
          tangent-envelope heat-smoothing marginal-divide

all: $(FIGURES:%=../figures/%.png) ../featured.png

../featured.png: quantile-identity.tex
	xelatex -interaction=nonstopmode quantile-identity.tex
	pdftoppm -png -r 350 -singlefile quantile-identity.pdf quantile-identity
	mv quantile-identity.png $@

../figures/%.png: %.tex
	xelatex -interaction=nonstopmode $<
	pdftoppm -png -r 350 -singlefile $*.pdf $*
	mv $*.png $@

clean:
	rm -f *.aux *.log *.pdf

.PHONY: all clean
