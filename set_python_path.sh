if [[ "$#" -ge "1" ]]; then
    BUILD_DIR="$1"
else
    BUILD_DIR="build"
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "The setpath.sh script must be sourced, not executed.\n"
    echo "$ source setpath.sh\n"
    exit 0
fi

if [ "$BASH_VERSION" ]; then
    LUISA_PATH=$(dirname "$BASH_SOURCE")
    export LUISA_PATH=$(builtin cd "$LUISA_PATH"; builtin pwd)
elif [ "$ZSH_VERSION" ]; then
    export LUISA_PATH=$(dirname "$0:A")
fi

export PYTHONPATH="$LUISA_PATH/$BUILD_DIR/bin:$PYTHONPATH"
export PATH="$LUISA_PATH/$BUILD_DIR/bin:$PATH"