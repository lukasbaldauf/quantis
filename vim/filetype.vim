if exists("did_load_filetypes")
   finish
endif
augroup filetypedetect
   au! BufNewFile,BufRead *.inp setf cp2k
augroup END

augroup filetypedetect
   au! BufNewFile,BufRead *.input setf tcl
augroup END

au BufRead,BufNewFile *.toml set filetype=toml
