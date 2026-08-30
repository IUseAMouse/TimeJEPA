# Build the TimeJEPA technical report.
# Either a full TeX Live (pdflatex+bibtex) or tectonic works:
#   make            -> tectonic if available, else pdflatex chain
#   make pdflatex   -> classic 4-pass build
# grep PENDING sections/*.tex to find spots awaiting running results.

TECTONIC := $(shell command -v tectonic 2>/dev/null)

all:
ifdef TECTONIC
	tectonic main.tex
else
	$(MAKE) pdflatex
endif

pdflatex:
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.xdv

.PHONY: all pdflatex clean
