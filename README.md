# AutoLALA

## How to Compile

(assume ubuntu)
```bash
# install LLVM
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 20

# install build tools
sudo apt install build-essential cmake autoconf libtool

# set environment variables
export MLIR_SYS_200_PREFIX=/usr/lib/llvm-20
export TABLEGEN_200_PREFIX=/usr/lib/llvm-20

# build and test
cargo build --release
cargo test --release
```

## Recommended development setup (for VSCode)

- Install DirEnv 
  - [Installation](https://direnv.net/docs/installation.html)
  - [Setup](https://direnv.net/docs/hook.html)

- Create `.envrc` file at the root of this project
  ```bash
  # .envrc
  export MLIR_SYS_200_PREFIX=/usr/lib/llvm-20
  export TABLEGEN_200_PREFIX=/usr/lib/llvm-20
  ```

- Install `rust-analyzer` extension for VSCode
  - [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=matklad.rust-analyzer)

  Add the following settings to `.vscode/settings` under the root of this project
  ```json
  {
    "rust-analyzer.cargo.extraEnv": {
        "MLIR_SYS_200_PREFIX" : "/usr/lib/llvm-20",
        "TABLEGEN_200_PREFIX" : "/usr/lib/llvm-20",
    },
    "rust-analyzer.check.extraEnv": {
        "MLIR_SYS_200_PREFIX" : "/usr/lib/llvm-20",
        "TABLEGEN_200_PREFIX" : "/usr/lib/llvm-20",
    },
    "rust-analyzer.server.extraEnv": {
        "MLIR_SYS_200_PREFIX" : "/usr/lib/llvm-20",
        "TABLEGEN_200_PREFIX" : "/usr/lib/llvm-20",
    },
    "rust-analyzer.runnables.extraEnv": {
        "MLIR_SYS_200_PREFIX" : "/usr/lib/llvm-20",
        "TABLEGEN_200_PREFIX" : "/usr/lib/llvm-20",
    },
    "editor.formatOnSave": true,
    "files.insertFinalNewline": true,
  }
  ```
