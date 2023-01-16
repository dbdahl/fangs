set shell := ["fish", "-c"]
package := `Rscript -e "cat(read.dcf('"(fd DESCRIPTION)"')[,'Package'])"`
version := `Rscript -e "cat(read.dcf('"(fd DESCRIPTION)"')[,'Version'])"`

install: fast-install

fast-install:
    R CMD INSTALL {{package}}

prebuild:
    cd {{package}}; Rscript -e 'cargo::prebuild("all")'

build: prebuild
    R CMD build {{package}}

check: build
    R CMD check --as-cran {{package}}_{{version}}.tar.gz

full-install: build
    R CMD INSTALL {{package}}_{{version}}.tar.gz

