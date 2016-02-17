ALL=	qso-templates.pdf

LATEX = pdflatex
BIBTEX = bibtex

%.pdf: %.tex
	$(LATEX) $*
#	$(BIBTEX) $*
#	$(LATEX) $*
	$(LATEX) $*
	$(LATEX) $*

all: $(ALL)

clean:
	rm -f *.log *.dvi *.aux *.blg *.ps

dummy:
