#!/bin/bash



rm -rf *.aux
rm -rf *.bbl
rm -rf *.bcf
rm -rf *.blg
rm -rf *.idx
rm -rf *.ilg
rm -rf *.ind
rm -rf *.log
rm -rf *.out
rm -rf *.toc
rm -rf *.xml
rm -rf *.xml
rm -rf *.xml
rm -rf *.xml
rm -rf *.xml

lualatex -pdf -lualatex main.tex

