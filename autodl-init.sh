if grep -q "# autodl-init-done" ~/.bashrc; then
    exit 0
else
    echo "# autodl-init-done
    " >> ~/.bashrc
fi

# 便携指令
echo '
alias net-turbo='\''
if [[ -z "$https_proxy" && -z "$http_proxy" && -z "$all_proxy" ]]; then
    export https_proxy="http://172.16.16.117:20171"
    export http_proxy="http://172.16.16.117:20171" 
    export all_proxy="socks5://172.16.16.117:20170"
    echo "外网资源代理已启动(Sponsored by Trevor.Z)"
else
    unset https_proxy
    unset http_proxy
    unset all_proxy
    echo "外网资源代理已关闭"
fi'\''

alias tmux-s='\''tmux new -s'\''
alias tmux-r='\''tmux attach -t'\''   # 让 tmux -r 也能用
alias tmux-ls='\''tmux ls'\''
alias tmux-rm='\''tmux kill-session -t'\''

eval "$(uv generate-shell-completion bash)"
curl -sSf https://cdn.puluter.cn/f/bash/notify.sh | bash
' >> ~/.bashrc


echo 'set -g mouse on
set -g history-limit 100000
set -g history-file ~/.tmux_history' >> ~/.tmux.conf

bash <(curl -sSL https://linuxmirrors.cn/main.sh) --source mirror.iscas.ac.cn --branch ubuntu --upgrade-software false --protocol https --clean-cache true 

# 换源 pip
pip config set global.index-url https://mirrors.cernet.edu.cn/pypi/web/simple

# 安装依赖
apt-get install -y nano swig tmux
pip install py3nvml uv ruff wandb

source ~/.bashrc
net-turbo

eval "$(curl https://get.x-cmd.com)"
x theme use pl-hg/orange-1
