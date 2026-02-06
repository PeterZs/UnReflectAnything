_unreflectanything() {
    local -a subcommands
    subcommands=(train test inference sweep agent completion download verify verify cite)
    _arguments -C \
        '(-h --help)'{-h,--help}'[Show help and exit]' \
        '1:subcommand:($subcommands)' \
        '*::args: _normal'
}

(( $+functions[_unreflectanything] )) && compdef _unreflectanything unreflectanything unreflect ura
