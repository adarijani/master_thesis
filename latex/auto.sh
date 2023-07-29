#!/bin/bash

git pull
lualatex thesis-template.tex
#convert thesis-template.pdf thesis-template.jpg
#convert thesis-template.jpg -resize 30%  thesis-template-30.jpg
git add .
git commit -a -m "commit" 
git push
