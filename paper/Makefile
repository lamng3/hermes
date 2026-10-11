.PHONY: all clean

all: hermes.pdf

hermes.pdf: hermes.tex hermes.bib acmart.cls pvldb.sty ACM-Reference-Format.bst
	pdflatex -interaction=nonstopmode hermes
	bibtex hermes
	pdflatex -interaction=nonstopmode hermes
	pdflatex -interaction=nonstopmode hermes

clean:
	rm -f hermes.aux hermes.bbl hermes.blg hermes.log hermes.out \
	      hermes.fdb_latexmk hermes.fls hermes.synctex.gz
