.PHONY: all clean

all: hermes.pdf

# Uses pdflatex and bibtex when installed, otherwise tectonic.
hermes.pdf: hermes.tex hermes.bib acmart.cls pvldb.sty ACM-Reference-Format.bst
	@if command -v pdflatex >/dev/null; then \
	  pdflatex -interaction=nonstopmode hermes && bibtex hermes && \
	  pdflatex -interaction=nonstopmode hermes && pdflatex -interaction=nonstopmode hermes; \
	else tectonic hermes.tex; fi

clean:
	rm -f hermes.aux hermes.bbl hermes.blg hermes.log hermes.out \
	      hermes.fdb_latexmk hermes.fls hermes.synctex.gz
