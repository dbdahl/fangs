# An enhancement of the system2 function which sets environment variables better.
system3 <- function(..., env=character()) {
  if ( length(env) > 0 ) {
    names <- names(env)
    original_env <- sapply(names, function(x) Sys.getenv(x,"<unset>"))
    tmp <- original_env != "<unset>"
    to_restore <- original_env[tmp]
    to_unset <- names(original_env[!tmp])
    tmp <- ! is.na(env)
    if ( sum(tmp) > 0 ) do.call(Sys.setenv, as.list(env[tmp]))
    if ( sum(!tmp) > 0 ) Sys.unsetenv(names(env[!tmp]))
    on.exit({
      if ( length(to_restore) > 0 ) do.call(Sys.setenv, as.list(to_restore))
      Sys.unsetenv(to_unset)
    })
  }
  system2(...)
}

get_homes <- function(cargo_home, rustup_home) {
  c(CARGO_HOME=normalizePath(cargo_home, mustWork=FALSE),
    RUSTUP_HOME=normalizePath(rustup_home, mustWork=FALSE))
}

mk_rustflags <- function(...) {
  args <- c(...)
  if ( is.null(args) ) list()
  else {
    x <- if ( any(is.na(args)) || (length(args) == 0) ) {
      NA
    } else {
      paste(args, collapse=rawToChar(as.raw(strtoi("1f",base=16L))))
    }
    list(CARGO_ENCODED_RUSTFLAGS=x)
  }
}

run <- function(..., minimum_version=".", methods=c("envir","path","cache"), environment_variables=list(), rustflags=NULL, use_packageStartupMessage=FALSE, must_be_silent=FALSE) {
  args <- shQuote(c(...))
  msg <- function(...) {
    if ( must_be_silent ) return()
    if ( use_packageStartupMessage ) {
      packageStartupMessage(..., appendLF=FALSE)
    } else {
      base::message(..., appendLF=FALSE)
    }
  }
  desc_file <- file.path(minimum_version, "DESCRIPTION")
  msrv <- if ( file.exists(desc_file) ) {
    desc <- read.dcf(desc_file)
    x <- tryCatch(as.character(desc[,"SystemRequirements"]), error=function(e) NA)
    if ( is.na(x) ) {
      msg("Could not find 'SystemRequirements' field in DESCRIPTION file.\n")
      return(100)
    }
    y <- gsub(".*[Cc]argo\\s*\\(>=\\s*([^)]+)\\).*","\\1", x)
    if ( identical(x,y) ) {
      msg("Could not find expected 'SystemRequirements: Cargo (>= XXXX)' in DESCRIPTION file.")
      return(101)
    }
    gsub("\\s*([^ ]+)\\s*","\\1",y)
  } else minimum_version
  check_candidate <- function(cargo_home, rustup_home, can_install=FALSE, can_update=FALSE) {
    vars <- c(get_homes(cargo_home, rustup_home), mk_rustflags(rustflags), environment_variables)
    cargo_cmd <- file.path(cargo_home, "bin", paste0("cargo", ifelse(windows,".exe","")))
    if ( ! file.exists(cargo_cmd) ) {
      if ( ! can_install ) return(201)
      if ( ! install_engine(FALSE, use_packageStartupMessage, must_be_silent) ) {
        return(202)
      }
    }
    output <- system3(cargo_cmd, "--version", stdout=TRUE, env=vars)
    if ( ! is.null(attr(output,"status")) ) {
      msg("Cargo is installed, but broken.\nPlease try again by running 'cargo::install()' in an interactive session.\n")
      return(203)
    }
    version <- tryCatch({
      version <- strsplit(output," ",fixed=TRUE)[[1]][2]
      if ( is.na(version) ) {
        msg(sprintf("Problem parsing Cargo version string: '%s'.\nPlease try again by running 'cargo::install()' in an interactive session.\n",paste(output,collapse=",")))
        return(204)
      }
      if ( utils::compareVersion(version, msrv) < 0 ) {
        msg(sprintf("Cargo version '%s' is installed, but '%s' is needed. Trying to update...\n",version,msrv))
        if ( ! can_update ) {
          msg("Upgrading this version is not permitted.\n")
          return(205)
        }
        rustup_cmd <- file.path(cargo_home, "bin", paste0("rustup", ifelse(windows,".exe","")))
        exit_status <- system3(rustup_cmd, "update", env=vars)
        if ( exit_status != 0 ) {
          msg("Upgrade failed.\nPlease try again by running 'cargo::install()' in an interactive session.\n")
          return(exit_status)
        }
      } else {
        msg(sprintf("Cargo version '%s' is installed, which satisfies the need for '%s'.\n",version,msrv))
      }
      version
    }, warning=identity, error=identity)
    if ( inherits(version,"warning") || inherits(version,"error") ) {
      msg(sprintf("Problem parsing Cargo version string '%s', comparing it against '%s', or running 'rustup update'.\nPlease try again by running 'cargo::install()' in an interactive session.\n",paste(output,collapse=","),msrv))
      return(206)
    }
    0
  }
  windows <- .Platform$OS.type=="windows"
  run_engine <- function(bypass_env_var, condition, cargo_home, rustup_home, can_install, can_update) {
    general_bypass_env_var <- "R_CARGO_RUN"
    if ( toupper(Sys.getenv(general_bypass_env_var, "TRUE")) == "FALSE" ) {
      msg(sprintf("Method bypassed by %s environment variable.\n", general_bypass_env_var))
      return(NULL)
    }
    if ( toupper(Sys.getenv(bypass_env_var, "TRUE")) == "FALSE" ) {
      msg(sprintf("Method bypassed by %s environment variable.\n", bypass_env_var))
      return(NULL)
    }
    if ( condition ) {
      status <- check_candidate(cargo_home, rustup_home, can_install, can_update)
      if ( status == 0 ) {
        cargo_cmd <- file.path(cargo_home, "bin", paste0("cargo", ifelse(windows,".exe","")))
        msg(sprintf("Cargo found at: %s\n", cargo_cmd))
        vars <- c(get_homes(cargo_home, rustup_home), mk_rustflags(rustflags), environment_variables)
        return(system3(cargo_cmd, args, env=vars))
      } else {
        msg("Method failed.\n")
      }
    } else {
      msg("Condition not met.\n")
    }
    NULL
  }
  for ( method in methods ) {
    if ( method == "envir" ) {
      msg("Trying to find a suitable Cargo using the CARGO_HOME and RUSTUP_HOME environment variables.\n")
      status <- run_engine(
        "R_CARGO_RUN_ENVIR",
        ( Sys.getenv("CARGO_HOME","<unset>") != "<unset>" ) && ( Sys.getenv("RUSTUP_HOME","<unset>") != "<unset>" ),
        Sys.getenv("CARGO_HOME"), Sys.getenv("RUSTUP_HOME"), FALSE, FALSE)
      if ( ( ! is.null(status) ) && ( status == 0 ) ) return(status)
    } else if ( method == "path" ) {
      prefix <- ifelse(windows,"%USERPROFILE%","$HOME")
      msg(sprintf("Trying to find a suitable Cargo at %s/.cargo and %s/.rustup.\n", prefix, prefix))
      prefix_dir <- Sys.getenv(ifelse(windows,"USERPROFILE","HOME"))
      status <- run_engine(
        "R_CARGO_RUN_PATH",
        Sys.getenv(ifelse(windows,"USERPROFILE","HOME"),"<unset>") != "<unset>",
        file.path(prefix_dir, ".cargo"), file.path(prefix_dir, ".rustup"), FALSE, FALSE)
      if ( ( ! is.null(status) ) && ( status == 0 ) ) return(status)
    } else if ( method == "cache" ) {
      prefix_dir <- tools::R_user_dir("cargo", "cache")
      msg("Trying to find a suitable Cargo using tools::R_user_dir('cargo', 'cache').\n")
      status <- run_engine(
        "R_CARGO_RUN_CACHE",
        TRUE,
        file.path(prefix_dir, "cargo"), file.path(prefix_dir, "rustup"), TRUE, TRUE)
      if ( ( ! is.null(status) ) && ( status == 0 ) ) return(status)
    }
  }
  msg("Cargo not found.\n")
  100
}

install_engine <- function(force, use_packageStartupMessage, must_be_silent) {
    msg <- function(...) {
        if ( must_be_silent ) return()
        if ( isTRUE(use_packageStartupMessage) ) {
            packageStartupMessage(..., appendLF=FALSE)
        } else {
            base::message(..., appendLF=FALSE)
        }
    }
    cache_dir <- tools::R_user_dir("cargo", "cache")
    days_until_next_purge <- 91
    last_purge_filename <- file.path(cache_dir,"last-purge")
    message <- sprintf(
'\nThe cargo package would like to download Rust from https://rustup.rs/ (an
official website of the Rust project) and then install Rust into the directory:
    %s
That directory will then be used to keep the Rust installation up-to-date. It
will also be used to: 1. cache shared libraries for R packages based on Rust
and 2. enable cached compilation for the cargo::rust_fn function. The cargo
package purges unused cache items every %s days, but you can change the
frequency by modifying the last line of the "%s" file in that
directory. You can revoke permission at any time by deleting that directory.\n\n',
cache_dir, days_until_next_purge, basename(last_purge_filename))
    if ( isFALSE(force) ) {
        if ( must_be_silent || use_packageStartupMessage || ! interactive() ) {
            msg("Please try again in an interactive session or use 'cargo::install(force=TRUE)'.\n")
            return(invisible(FALSE))
        }
        while ( TRUE ) {
            msg(message)
            msg("Do you agree? [y/N] ")
            response <- toupper(trimws(readline()))
            if ( response %in% c("N","") ) return(invisible(FALSE))
            if ( response %in% c("Y") ) break
            msg("\n")
        }
        msg("Proceeding with installation. Please be patient.\n")
    } else {
        msg(message)
        msg("Agreement accepted due to 'force=TRUE'.\n")
    }
    msg("\n")
    cargo_home  <- file.path(cache_dir,"cargo")
    rustup_home <- file.path(cache_dir,"rustup")
    vars <- get_homes(cargo_home, rustup_home)
    if ( unlink(vars, recursive=TRUE, force=TRUE) != 0 ) {
        msg(sprintf("Could not clean out installation directory:\n    %s\nPlease delete this directory.\n",normalizePath(cache_dir)))
        return(invisible(FALSE))
    }
    dir.create(cache_dir, showWarnings=FALSE, recursive=TRUE)
    windows <- .Platform$OS.type=="windows"
    rustup_init <- file.path(cache_dir, sprintf("rustup-init.%s",ifelse(windows,"exe","sh")))
    URL <- ifelse(windows,"https://win.rustup.rs/x86_64","https://sh.rustup.rs")
    if ( tryCatch(utils::download.file(URL, rustup_init, mode="wb", quiet=use_packageStartupMessage), warning=function(e) 1, error=function(e) 1) != 0 ) {
        msg(sprintf("Could not download '%s' to '%s'.\nPlease try again by running 'cargo::install()' in an interactive session.\n", URL, rustup_init))
        return(invisible(FALSE))
    }
    msg("Running installation. Please be patient.\n")
    rustup_init_stdout <- file.path(cache_dir, "rustup-init.stdout")
    rustup_init_stderr <- file.path(cache_dir, "rustup-init.stderr")
    if ( windows ) {
        lines <- paste0(shQuote(normalizePath(rustup_init, mustWork=FALSE)), " --no-modify-path -y --default-host x86_64-pc-windows-gnu")
        rustup_init_bat <- file.path(cache_dir, "rustup-init.bat")
        writeLines(lines, rustup_init_bat)
        status <- system3(rustup_init_bat, stdout=rustup_init_stdout, stderr=rustup_init_stderr, env=vars)
        if ( status != 0 ) {
            msg(sprintf("There was a problem running the rustup installer at '%s'.\nSee '%s' and '%s'.\nPlease try again by running 'cargo::install()' in an interactive session.\n", rustup_init, rustup_init_stdout, rustup_init_stderr))
            return(invisible(FALSE))
        }
        unlink(rustup_init_bat)
    } else {
        if ( system3("sh", c(shQuote(rustup_init),"--no-modify-path","-y"), stdout=rustup_init_stdout, stderr=rustup_init_stderr, env=vars) != 0 ) {
            msg(sprintf("There was a problem running the rustup installer at '%s'.\nSee '%s' and '%s'.\nPlease try again by running 'cargo::install()' in an interactive session.\n", rustup_init, rustup_init_stdout, rustup_init_stderr))
            return(invisible(FALSE))
        }
    }
    unlink(rustup_init)
    unlink(rustup_init_stdout)
    unlink(rustup_init_stderr)
    writeLines(c("1",as.character(Sys.Date()),days_until_next_purge), last_purge_filename)
    msg("Installation was successfull.\n")
    invisible(TRUE)
}
