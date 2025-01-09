

FROM mirror-registry.cn-hangzhou.cr.aliyuncs.com/yuxian/base-devel:base-devel  AS libriichi_build

RUN <<EOF
pacman -Syu --noconfirm --needed rust python
pacman -Scc
EOF

WORKDIR ./
COPY Cargo.toml Cargo.lock .
COPY libriichi libriichi
COPY exe-wrapper exe-wrapper

RUN sudo pacman -S --needed --noconfirm base-devel git 
#RUN  git clone https://aur.archlinux.org/pyenv.git
#RUN cd pyenv && makepkg -si && cd ../
RUN curl https://pyenv.run | bash
RUN cat <<EOL >> ~/.bashrc
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
EOL
RUN source ~/.bashrc && pyenv install 3.10.0 && pyenv global 3.10.0
ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
#ENV PATH="/usr/local/bin:/usr/bin:/usr/local/games:/usr/lib/python3.12"
RUN cargo build -p libriichi --lib --release
# RUN cargo build -p exe-wrapper --release
# -----
# FROM archlinux:base
FROM mirror-registry.cn-hangzhou.cr.aliyuncs.com/yuxian/archlinux:base

RUN <<EOF
pacman -Syu --noconfirm python python-pip
pacman -S base-devel openssl zlib xz
pacman -S pyenv
pyenv install 3.10.0
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc 
pyenv global 3.10.0
python -m venv pytorch_env
source pytorch_env/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install toml tqdm tensorboard

#pacman -Syu --noconfirm --needed python python-pytorch-cuda python-toml python-tqdm tensorboard
pacman -Scc
EOF

WORKDIR ./mortal
COPY mortal .
COPY --from=libriichi_build /target/release/libriichi.so .

ENV MORTAL_CFG config.toml
COPY <<'EOF' config.toml
[control]
state_file = '/mnt/mortal.pth'

[resnet]
conv_channels = 192
num_blocks = 40
enable_bn = true
bn_momentum = 0.99
EOF

VOLUME /mnt

CMD ["/bin/bash"]
#ENTRYPOINT ["python", "mortal.py"]
