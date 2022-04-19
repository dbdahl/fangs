set shell := ["fish", "-c"]

package := `echo (basename (R-package path))_(R-package version).tar.gz`

notes:
    R-package notes

build:
    R-package build

check: build
    R CMD check --as-cran {{package}}

deploy: build && tag
    R-package cranlike {{package}}
    R-package deploy --cranlike

tag:
    git isclean
    R-package tag

