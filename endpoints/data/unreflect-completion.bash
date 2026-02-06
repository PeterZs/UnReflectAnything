# Bash completion for unreflectanything / unreflect / ura
# Source: source <(unreflectanything completion bash)

_unreflectanything() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local subcommands="train test inference sweep agent completion download-weights"
    local options="-h --help"
    COMPREPLY=($(compgen -W "$subcommands $options" -- "$cur"))
}

complete -F _unreflectanything unreflectanything unreflect ura
