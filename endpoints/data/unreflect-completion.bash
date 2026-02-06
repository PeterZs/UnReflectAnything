_unreflectanything() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local subcommands="train test inference sweep agent completion download verify verify cite"
    local options="-h --help"
    COMPREPLY=($(compgen -W "$subcommands $options" -- "$cur"))
}

complete -F _unreflectanything unreflectanything unreflect ura
