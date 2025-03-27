figures: images/qp-sim-cost.png

images:
	mkdir -p images

images/%.png : figures/%.dot images
	dot $< -Tpng -o $@

images/%.svg : figures/%.dot images
	dot $< -Tsvg -o $@

.PHONY: clean
clean:  ## remove all generated files
	rm -f $(wildcard images/*.png)
