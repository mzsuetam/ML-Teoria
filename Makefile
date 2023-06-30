pdf: ml-teoria.md
	pandoc ml-teoria.md -o ml-teoria.pdf \
		--pdf-engine=xelatex \
		--metadata-file=config.yaml \
		--toc \
		--number-sections
# -V lang=pl 

autosave:
	python autosave.py