#!/bin/sh

device=false
current_dir="$(pwd -P)"

check_requirements() {
    case $(uname -s) in
        Darwin)
            if [ "$(uname -m)" = "arm64" ]; then
                printf "macOS (Apple Silicon) system detected.\n"
                device="osx-arm64"
            else
                printf "macOS (Intel) system detected.\n"
                export CFLAGS='-stdlib=libc++'
                device="osx-64"
            fi
            ;;
        Linux)
            printf "Linux system detected.\n"
            device="linux-64"
            ;;
        *)
            printf "Only Linux and macOS are currently supported.\n"
            exit 1
            ;;
    esac
}


install_packages() {
    printf "Installing required packages...\n"
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        printf "requirements.txt file not found. Unable to install packages.\n"
        exit 1
    fi
}

check_requirements
install_packages

printf '\nSetup completed.\n'
