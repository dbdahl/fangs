#![allow(dead_code)]
#![allow(clippy::wrong_self_convention)]

// See:
//   https://cran.r-project.org/doc/manuals/r-release/R-ints.html
//   https://svn.r-project.org/R/trunk/src/include/Rinternals.h
//   https://github.com/hadley/r-internals

use crate::rbindings::*;
use std::convert::{TryFrom, TryInto};
use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_int};

// See https://doc.rust-lang.org/nomicon/unwinding.html
//
// There is an API called catch_unwind that enables catching a panic without spawning a thread. Still, we would encourage you to only do this sparingly. In particular, Rust's current unwinding implementation is heavily optimized for the "doesn't unwind" case. If a program doesn't unwind, there should be no runtime cost for the program being ready to unwind. As a consequence, actually unwinding will be more expensive than in e.g. Java. Don't build your programs to unwind under normal circumstances. Ideally, you should only panic for programming errors or extreme problems.

#[macro_export]
macro_rules! panic_to_error {
    ($blk: block) => {{
        let result = std::panic::catch_unwind(|| $blk);
        match result {
            Ok(obj) => obj,
            Err(e) => crate::r::error("Panic in Rust!"),
        }
    }};
}

pub fn error(message: &str) -> SEXP {
    unsafe {
        let list = list_vector_with_names_and_values(&[
            ("message", string_scalar(message).protect()),
            ("call", R_NilValue),
        ])
        .protect();
        Rf_classgets(list, string_vector_with_values(&["error", "condition"]));
        unprotect(2);
        list
    }
}

// Use to seed a RNG based on R's RNG state.
pub fn random_bytes<const LENGTH: usize>() -> [u8; LENGTH] {
    unsafe {
        let m = (u8::MAX as f64) + 1.0;
        let mut bytes: [u8; LENGTH] = [0; LENGTH];
        GetRNGstate();
        for x in bytes.iter_mut() {
            *x = R_unif_index(m) as u8;
        }
        PutRNGstate();
        bytes
    }
}

// Use to seed a RNG based on R's RNG state.
// Keep for demonstration purposes only.
pub fn random_bytes_demonstration<const LENGTH: usize>() -> [u8; LENGTH] {
    unsafe {
        let result = Rf_install(b"sample.int\0".as_ptr() as *const c_char)
            .protect()
            .call2(
                integer_scalar((u8::MAX as c_int) + 1).protect(),
                integer_scalar(LENGTH as c_int).protect(),
            )
            .protect();
        let slice = result.as_integer_slice();
        let mut bytes: [u8; LENGTH] = [0; LENGTH];
        bytes
            .iter_mut()
            .zip(slice)
            .for_each(|(b, s)| *b = (*s - 1) as u8);
        unprotect(4);
        bytes
    }
}

pub fn print(x: &str) {
    unsafe {
        Rprintf(
            b"%.*s\0".as_ptr() as *const c_char,
            x.len(),
            x.as_ptr() as *const c_char,
        );
    }
}

pub fn unprotect(x: c_int) {
    unsafe {
        Rf_unprotect(x);
    }
}

pub fn nil() -> SEXP {
    unsafe { R_NilValue }
}

pub fn logical_scalar(x: bool) -> SEXP {
    unsafe { Rf_ScalarLogical(x.into()) }
}

pub fn integer_scalar(x: c_int) -> SEXP {
    unsafe { Rf_ScalarInteger(x) }
}

pub fn double_scalar(x: c_double) -> SEXP {
    unsafe { Rf_ScalarReal(x) }
}

pub fn character(x: &str) -> SEXP {
    unsafe {
        Rf_mkCharLen(
            x.as_ptr() as *const c_char,
            c_int::try_from(x.len()).unwrap(),
        )
    }
}

pub fn string_scalar(x: &str) -> SEXP {
    unsafe {
        let result = Rf_ScalarString(character(x).protect());
        unprotect(1);
        result
    }
}

fn mk_vector(tipe: u32, len: isize) -> SEXP {
    if len < 0 {
        panic!("Invalid length: {}", len);
    }
    unsafe { Rf_allocVector(tipe, len) }
}

pub fn logical_vector(len: isize) -> SEXP {
    mk_vector(LGLSXP, len)
}

pub fn integer_vector(len: isize) -> SEXP {
    mk_vector(INTSXP, len)
}

pub fn double_vector(len: isize) -> SEXP {
    mk_vector(REALSXP, len)
}

pub fn string_vector(len: isize) -> SEXP {
    mk_vector(STRSXP, len)
}

pub fn string_vector_with_values(x: &[&str]) -> SEXP {
    let len = x.len().try_into().unwrap();
    let y = string_vector(len).protect();
    for (i, x) in x.iter().enumerate() {
        unsafe { SET_STRING_ELT(y, i.try_into().unwrap(), character(*x)) };
    }
    unprotect(1);
    y
}

pub fn list_vector(len: isize) -> SEXP {
    mk_vector(VECSXP, len)
}

pub fn list_vector_with_names(names: &[&str]) -> SEXP {
    let list = list_vector(names.len().try_into().unwrap()).protect();
    let nms = string_vector_with_values(names);
    unsafe { Rf_namesgets(list, nms) };
    unprotect(2);
    list
}

pub fn list_vector_with_names_and_values(tuples: &[(&str, SEXP)]) -> SEXP {
    let len = tuples.len().try_into().unwrap();
    let list = mk_vector(VECSXP, len).protect();
    let names = string_vector(len).protect();
    for (i, x) in tuples.iter().enumerate() {
        let i = i.try_into().unwrap();
        let tuple = *x;
        unsafe {
            SET_STRING_ELT(names, i, character(tuple.0));
            SET_VECTOR_ELT(list, i, tuple.1);
        }
    }
    unsafe { Rf_namesgets(list, names) };
    unprotect(2);
    list
}

fn mk_matrix(tipe: u32, nrow: c_int, ncol: c_int) -> SEXP {
    if nrow < 0 {
        panic!("Invalid number of rows: {}", nrow);
    }
    if ncol < 0 {
        panic!("Invalid number of columns: {}", ncol);
    }
    unsafe { Rf_allocMatrix(tipe, nrow, ncol) }
}

pub fn logical_matrix(nrow: c_int, ncol: c_int) -> SEXP {
    mk_matrix(LGLSXP, nrow, ncol)
}

pub fn integer_matrix(nrow: c_int, ncol: c_int) -> SEXP {
    mk_matrix(INTSXP, nrow, ncol)
}

pub fn double_matrix(nrow: c_int, ncol: c_int) -> SEXP {
    mk_matrix(REALSXP, nrow, ncol)
}

pub fn string_matrix(nrow: c_int, ncol: c_int) -> SEXP {
    mk_matrix(STRSXP, nrow, ncol)
}

fn mk_dim_protected(dim: &[c_int]) -> SEXP {
    let dim2 = integer_vector(R_xlen_t::try_from(dim.len()).unwrap()).protect();
    fn m(x: &c_int) -> c_int {
        let y = *x;
        if y < 0 {
            panic!("Invalid dimension: {}", y);
        }
        y
    }
    dim2.fill_integer_from(dim, m);
    dim2
}

pub fn logical_array(dim: &[c_int]) -> SEXP {
    let result = unsafe { Rf_allocArray(LGLSXP, mk_dim_protected(dim)) };
    unprotect(1);
    result
}

pub fn integer_array(dim: &[c_int]) -> SEXP {
    let result = unsafe { Rf_allocArray(INTSXP, mk_dim_protected(dim)) };
    unprotect(1);
    result
}

pub fn double_array(dim: &[c_int]) -> SEXP {
    let result = unsafe { Rf_allocArray(REALSXP, mk_dim_protected(dim)) };
    unprotect(1);
    result
}

pub fn string_array(dim: &[c_int]) -> SEXP {
    let result = unsafe { Rf_allocArray(STRSXP, mk_dim_protected(dim)) };
    unprotect(1);
    result
}

pub trait SEXPExt {
    fn protect(self) -> SEXP;
    fn duplicate(self) -> SEXP;
    fn is_logical(self) -> bool;
    fn is_integer(self) -> bool;
    fn is_double(self) -> bool;
    fn is_scalar_integer_or_double(self) -> bool;
    fn is_raw(self) -> bool;
    fn is_character(self) -> bool;
    fn is_string(self) -> bool;
    fn is_list(self) -> bool;
    fn is_vector_atomic(self) -> bool;
    fn is_matrix(self) -> bool;
    fn is_array(self) -> bool;
    fn is_nil(self) -> bool;
    fn is_function(self) -> bool;
    fn is_environment(self) -> bool;
    fn as_logical(self) -> c_int;
    fn as_bool(self) -> bool;
    fn as_integer(self) -> c_int;
    fn as_usize(self) -> usize;
    fn as_double(self) -> c_double;
    fn as_string(self) -> &'static str;
    fn as_logical_slice_mut(self) -> &'static mut [c_int];
    fn as_logical_slice(self) -> &'static [c_int];
    fn as_integer_slice_mut(self) -> &'static mut [c_int];
    fn as_integer_slice(self) -> &'static [c_int];
    fn as_double_slice_mut(self) -> &'static mut [c_double];
    fn as_double_slice(self) -> &'static [c_double];
    fn as_raw_slice_mut(self) -> &'static mut [u8];
    fn as_raw_slice(self) -> &'static [u8];
    fn to_integer(self) -> SEXP;
    fn to_double(self) -> SEXP;
    fn fill_logical_from<T>(self, slice: &[T], mapper: fn(&T) -> c_int);
    fn fill_integer_from<T>(self, slice: &[T], mapper: fn(&T) -> c_int);
    fn fill_double_from<T>(self, slice: &[T], mapper: fn(&T) -> c_double);
    fn fill_raw_from<T>(self, slice: &[T], mapper: fn(&T) -> u8);
    fn get_list_element(self, i: isize) -> SEXP;
    fn set_list_element(self, i: isize, value: SEXP);
    fn get_string_element(self, i: isize) -> SEXP;
    fn set_string_element(self, i: isize, value: &str);
    fn set_string_element_with_character(self, i: isize, value: SEXP);
    fn length(self) -> c_int;
    fn length_usize(self) -> usize;
    fn xlength(self) -> R_xlen_t;
    fn xlength_usize(self) -> usize;
    fn nrow(self) -> c_int;
    fn nrow_usize(self) -> usize;
    fn ncol(self) -> c_int;
    fn ncol_usize(self) -> usize;
    fn call0(self) -> SEXP;
    fn call1(self, x1: SEXP) -> SEXP;
    fn call2(self, x1: SEXP, x2: SEXP) -> SEXP;
    fn call3(self, x1: SEXP, x2: SEXP, x3: SEXP) -> SEXP;
    fn call4(self, x1: SEXP, x2: SEXP, x3: SEXP, x4: SEXP) -> SEXP;
    fn call5(self, x1: SEXP, x2: SEXP, x3: SEXP, x4: SEXP, x5: SEXP) -> SEXP;
    fn call0_try(self) -> Option<SEXP>;
    fn call1_try(self, x1: SEXP) -> Option<SEXP>;
    fn call2_try(self, x1: SEXP, x2: SEXP) -> Option<SEXP>;
    fn call3_try(self, x1: SEXP, x2: SEXP, x3: SEXP) -> Option<SEXP>;
    fn call4_try(self, x1: SEXP, x2: SEXP, x3: SEXP, x4: SEXP) -> Option<SEXP>;
    fn call5_try(self, x1: SEXP, x2: SEXP, x3: SEXP, x4: SEXP, x5: SEXP) -> Option<SEXP>;
}

fn try_call(expr: SEXP, env: SEXP) -> Option<SEXP> {
    let mut p_out_error: c_int = 0;
    let result = unsafe { R_tryEval(expr.protect(), env, &mut p_out_error as *mut c_int) };
    unprotect(1);
    match p_out_error {
        0 => Some(result),
        _ => None,
    }
}

impl SEXPExt for SEXP {
    fn protect(self) -> Self {
        unsafe { Rf_protect(self) }
    }
    fn duplicate(self) -> Self {
        unsafe { Rf_duplicate(self) }
    }
    fn is_logical(self) -> bool {
        unsafe { Rf_isLogical(self) != 0 }
    }
    fn is_integer(self) -> bool {
        unsafe { Rf_isInteger(self) != 0 }
    }
    fn is_double(self) -> bool {
        unsafe { Rf_isReal(self) != 0 }
    }
    fn is_scalar_integer_or_double(self) -> bool {
        (self.is_integer() || self.is_double()) && self.length() == 1
    }
    fn is_raw(self) -> bool {
        unsafe { TYPEOF(self) == RAWSXP.try_into().unwrap() }
    }
    fn is_character(self) -> bool {
        unsafe { TYPEOF(self) == CHARSXP.try_into().unwrap() }
    }
    fn is_string(self) -> bool {
        unsafe { Rf_isString(self) != 0 }
    }
    fn is_list(self) -> bool {
        unsafe { TYPEOF(self) == VECSXP.try_into().unwrap() }
    }
    fn is_vector_atomic(self) -> bool {
        unsafe { Rf_isVectorAtomic(self) != 0 }
    }
    fn is_matrix(self) -> bool {
        unsafe { Rf_isMatrix(self) != 0 }
    }
    fn is_array(self) -> bool {
        unsafe { Rf_isArray(self) != 0 }
    }
    fn is_nil(self) -> bool {
        unsafe { Rf_isNull(self) != 0 }
    }
    fn is_function(self) -> bool {
        unsafe { Rf_isFunction(self) != 0 }
    }
    fn is_environment(self) -> bool {
        unsafe { Rf_isEnvironment(self) != 0 }
    }
    fn as_logical(self) -> c_int {
        unsafe { Rf_asLogical(self) }
    }
    fn as_bool(self) -> bool {
        unsafe { Rf_asLogical(self) != 0 }
    }
    fn as_integer(self) -> c_int {
        unsafe { Rf_asInteger(self) }
    }
    fn as_usize(self) -> usize {
        usize::try_from(unsafe { Rf_asInteger(self) }).unwrap()
    }
    fn as_double(self) -> c_double {
        unsafe { Rf_asReal(self) }
    }
    fn as_string(self) -> &'static str {
        let sexp = if self.is_string() {
            if self.length() == 0 {
                panic!("Length must be at least one");
            }
            self.get_string_element(0)
        } else {
            self
        };
        if unsafe { TYPEOF(sexp) != CHARSXP.try_into().unwrap() } {
            panic!("Not a character or string");
        }
        let a = unsafe { R_CHAR(sexp) as *const c_char };
        let c_str = unsafe { CStr::from_ptr(a) };
        match c_str.to_str() {
            Ok(x) => x,
            Err(_) => panic!("Could not convert to UTF-8"),
        }
    }
    fn as_logical_slice_mut(self) -> &'static mut [c_int] {
        unsafe {
            if !self.is_logical() {
                panic!("Not logical data");
            }
            std::slice::from_raw_parts_mut(LOGICAL(self), self.xlength_usize())
        }
    }
    fn as_logical_slice(self) -> &'static [c_int] {
        if !self.is_logical() {
            panic!("Not logical data");
        }
        unsafe { std::slice::from_raw_parts(LOGICAL(self), self.xlength_usize()) }
    }
    fn as_integer_slice_mut(self) -> &'static mut [c_int] {
        unsafe {
            if !self.is_integer() {
                panic!("Not integer data");
            }
            std::slice::from_raw_parts_mut(INTEGER(self), self.xlength_usize())
        }
    }
    fn as_integer_slice(self) -> &'static [c_int] {
        if !self.is_integer() {
            panic!("Not integer data");
        }
        unsafe { std::slice::from_raw_parts(INTEGER(self), self.xlength_usize()) }
    }
    fn as_double_slice_mut(self) -> &'static mut [c_double] {
        if !self.is_double() {
            panic!("Not double data");
        }
        unsafe { std::slice::from_raw_parts_mut(REAL(self), self.xlength_usize()) }
    }
    fn as_double_slice(self) -> &'static [c_double] {
        if !self.is_double() {
            panic!("Not double data");
        }
        unsafe { std::slice::from_raw_parts(REAL(self), self.xlength_usize()) }
    }
    fn as_raw_slice_mut(self) -> &'static mut [u8] {
        if !self.is_raw() {
            panic!("Not raw data");
        }
        unsafe { std::slice::from_raw_parts_mut(RAW(self), self.xlength_usize()) }
    }
    fn as_raw_slice(self) -> &'static [u8] {
        if !self.is_raw() {
            panic!("Not raw data");
        }
        unsafe { std::slice::from_raw_parts(RAW(self), self.xlength_usize()) }
    }
    fn to_integer(self) -> SEXP {
        if self.is_double() | self.is_logical() | self.is_string() {
            unsafe { Rf_coerceVector(self, INTSXP) }
        } else if self.is_integer() {
            self
        } else {
            panic!("Not an integer nor a double vector.")
        }
    }
    fn to_double(self) -> SEXP {
        if self.is_integer() | self.is_logical() | self.is_string() {
            unsafe { Rf_coerceVector(self, REALSXP) }
        } else if self.is_double() {
            self
        } else {
            panic!("Not an integer nor a double vector.")
        }
    }
    fn fill_logical_from<T>(self, src: &[T], mapper: fn(&T) -> c_int) {
        let dest = self.as_logical_slice_mut();
        for (a, b) in dest.iter_mut().zip(src.iter()) {
            *a = mapper(b);
        }
    }
    fn fill_integer_from<T>(self, src: &[T], mapper: fn(&T) -> c_int) {
        let dest = self.as_integer_slice_mut();
        for (a, b) in dest.iter_mut().zip(src.iter()) {
            *a = mapper(b);
        }
    }
    fn fill_double_from<T>(self, src: &[T], mapper: fn(&T) -> c_double) {
        let dest = self.as_double_slice_mut();
        for (a, b) in dest.iter_mut().zip(src.iter()) {
            *a = mapper(b);
        }
    }
    fn fill_raw_from<T>(self, src: &[T], mapper: fn(&T) -> u8) {
        let dest = self.as_raw_slice_mut();
        for (a, b) in dest.iter_mut().zip(src.iter()) {
            *a = mapper(b);
        }
    }
    fn get_list_element(self, i: isize) -> SEXP {
        if !self.is_list() {
            panic!("Not list");
        }
        unsafe { VECTOR_ELT(self, i) }
    }
    fn set_list_element(self, i: isize, value: SEXP) {
        if !self.is_list() {
            panic!("Not list");
        }
        unsafe {
            SET_VECTOR_ELT(self, i, value);
        }
    }
    fn get_string_element(self, i: isize) -> SEXP {
        if !self.is_string() {
            panic!("Not string");
        }
        unsafe { STRING_ELT(self, i) }
    }
    fn set_string_element(self, i: isize, value: &str) {
        if !self.is_string() {
            panic!("Not string");
        }
        unsafe {
            SET_STRING_ELT(self, i, character(value));
        }
    }
    fn set_string_element_with_character(self, i: isize, value: SEXP) {
        if !self.is_string() {
            panic!("Not string");
        }
        if !value.is_character() {
            panic!("Element is not character");
        }
        unsafe {
            SET_STRING_ELT(self, i, value);
        }
    }
    fn length(self) -> c_int {
        unsafe { Rf_length(self) }
    }
    fn length_usize(self) -> usize {
        usize::try_from(unsafe { Rf_length(self) }).unwrap()
    }
    fn xlength(self) -> R_xlen_t {
        unsafe { Rf_xlength(self) }
    }
    fn xlength_usize(self) -> usize {
        usize::try_from(unsafe { Rf_xlength(self) }).unwrap()
    }
    fn nrow(self) -> c_int {
        if !self.is_matrix() {
            panic!("Not a matrix");
        }
        unsafe { Rf_nrows(self) }
    }
    fn nrow_usize(self) -> usize {
        if !self.is_matrix() {
            panic!("Not a matrix");
        }
        usize::try_from(unsafe { Rf_nrows(self) }).unwrap()
    }
    fn ncol(self) -> c_int {
        if !self.is_matrix() {
            panic!("Not a matrix");
        }
        unsafe { Rf_ncols(self) }
    }
    fn ncol_usize(self) -> usize {
        if !self.is_matrix() {
            panic!("Not a matrix");
        }
        usize::try_from(unsafe { Rf_ncols(self) }).unwrap()
    }
    // Can long jump!
    fn call0(self) -> SEXP {
        unsafe {
            let result = Rf_eval(Rf_lang1(self).protect(), R_GetCurrentEnv());
            unprotect(1);
            result
        }
    }
    // Can long jump!
    fn call1(self, x1: SEXP) -> SEXP {
        unsafe {
            let result = Rf_eval(Rf_lang2(self, x1).protect(), R_GetCurrentEnv());
            unprotect(1);
            result
        }
    }
    // Can long jump!
    fn call2(self, x1: SEXP, x2: SEXP) -> SEXP {
        unsafe {
            let result = Rf_eval(Rf_lang3(self, x1, x2).protect(), R_GetCurrentEnv());
            unprotect(1);
            result
        }
    }
    // Can long jump!
    fn call3(self, x1: SEXP, x2: SEXP, x3: SEXP) -> SEXP {
        unsafe {
            let result = Rf_eval(Rf_lang4(self, x1, x2, x3).protect(), R_GetCurrentEnv());
            unprotect(1);
            result
        }
    }
    // Can long jump!
    fn call4(self, x1: SEXP, x2: SEXP, x3: SEXP, x4: SEXP) -> SEXP {
        unsafe {
            let result = Rf_eval(Rf_lang5(self, x1, x2, x3, x4).protect(), R_GetCurrentEnv());
            unprotect(1);
            result
        }
    }
    // Can long jump!
    fn call5(self, x1: SEXP, x2: SEXP, x3: SEXP, x4: SEXP, x5: SEXP) -> SEXP {
        unsafe {
            let result = Rf_eval(
                Rf_lang6(self, x1, x2, x3, x4, x5).protect(),
                R_GetCurrentEnv(),
            );
            unprotect(1);
            result
        }
    }
    fn call0_try(self) -> Option<SEXP> {
        unsafe { try_call(Rf_lang1(self), R_GetCurrentEnv()) }
    }
    fn call1_try(self, x1: SEXP) -> Option<SEXP> {
        unsafe { try_call(Rf_lang2(self, x1), R_GetCurrentEnv()) }
    }
    fn call2_try(self, x1: SEXP, x2: SEXP) -> Option<SEXP> {
        unsafe { try_call(Rf_lang3(self, x1, x2), R_GetCurrentEnv()) }
    }
    fn call3_try(self, x1: SEXP, x2: SEXP, x3: SEXP) -> Option<SEXP> {
        unsafe { try_call(Rf_lang4(self, x1, x2, x3), R_GetCurrentEnv()) }
    }
    fn call4_try(self, x1: SEXP, x2: SEXP, x3: SEXP, x4: SEXP) -> Option<SEXP> {
        unsafe { try_call(Rf_lang5(self, x1, x2, x3, x4), R_GetCurrentEnv()) }
    }
    fn call5_try(self, x1: SEXP, x2: SEXP, x3: SEXP, x4: SEXP, x5: SEXP) -> Option<SEXP> {
        unsafe { try_call(Rf_lang6(self, x1, x2, x3, x4, x5), R_GetCurrentEnv()) }
    }
}
