#!/bin/bash
# STEK 2035 Chatbot - Cloud (vault) setup + start script
# ==========================================================
# Run this in the JupyterLab terminal on compute.data-lab.site.
# Assumes both repos are already cloned into:
#   ~/vault/STEK2035-chatbot
#   ~/vault/STEK2035-Chatbot_FrontEnd/stek-chatbot
#
# Usage:
#   bash vault_setup.sh          # one-time install + start everything
#   source ~/vault/local/setup_env.sh   # in any LATER new session, instead
#                                        # of re-running this whole script

set -e

# --- 1. Node.js/npm into ~/vault (persists across container resets) ---
if [ ! -x "$HOME/vault/local/bin/node" ]; then
  echo "Installing Node.js into ~/vault/local ..."
  mkdir -p ~/vault/local
  cd /tmp
  curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz -o node.tar.xz
  tar -xJf node.tar.xz -C ~/vault/local --strip-components=1
  rm node.tar.xz
else
  echo "Node.js already installed in ~/vault/local, skipping."
fi

# --- 2. Python venv into ~/vault (persists) ---
if [ ! -d "$HOME/vault/local/venv" ]; then
  echo "Creating Python venv in ~/vault/local/venv ..."
  python3 -m venv ~/vault/local/venv
  source ~/vault/local/venv/bin/activate
  pip install -r ~/vault/STEK2035-chatbot/backend/requirements.txt
  pip install fastapi "uvicorn[standard]"
  pip install gensim spacy pyyaml pandas scikit-learn pymupdf sentence-transformers
  python -m spacy download de_core_news_sm
  python -m spacy download de_core_news_lg
else
  echo "Python venv already exists in ~/vault/local/venv, skipping install."
  source ~/vault/local/venv/bin/activate
fi

# --- 3. Persistent env script (source this every new session) ---
cat > ~/vault/local/setup_env.sh << 'EOF'
export PATH="$HOME/vault/local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/vault/local/lib/ollama:$LD_LIBRARY_PATH"
export OLLAMA_MODELS="$HOME/vault/local/models"
source "$HOME/vault/local/venv/bin/activate"
EOF
source ~/vault/local/setup_env.sh

# --- 4. Frontend deps ---
if [ ! -d "$HOME/vault/STEK2035-Chatbot_FrontEnd/stek-chatbot/node_modules" ]; then
  echo "Running npm install ..."
  npm install --prefix ~/vault/STEK2035-Chatbot_FrontEnd/stek-chatbot
else
  echo "node_modules already present, skipping npm install."
fi

# --- 5. Start Ollama (background) ---
pkill -f "ollama serve" 2>/dev/null || true
sleep 1
cd ~/vault/local && nohup ./bin/ollama serve > ollama.log 2>&1 &
sleep 2

# --- 6. Start backend (background) ---
cd ~/vault/STEK2035-chatbot
nohup uvicorn backend.rag_server:app --port 8000 > backend.log 2>&1 &
sleep 3

# --- 7. Start frontend (background) ---
nohup npm run dev --prefix ~/vault/STEK2035-Chatbot_FrontEnd/stek-chatbot > frontend.log 2>&1 &
sleep 3


# ==========================================================
# COMMANDS TO RUN AFTER THE SERVERS HAVE STARTED
# ==========================================================
echo ""
echo "=== Verifying everything is up ==="

echo "--- Ollama models ---"
ollama list

echo "--- Node/npm versions ---"
node -v && npm -v

echo "--- Python deps sanity check ---"
python -c "import fastapi, gensim, spacy; print('python deps ok')"

echo "--- Backend health check ---"
curl -s http://localhost:8000/health
echo ""

echo "--- Backend topics endpoint ---"
curl -s http://localhost:8000/topics
echo ""

echo "--- GPU check (see CLOUD_ENVIRONMENT_SETUP.txt if this errors) ---"
nvidia-smi || echo "nvidia-smi not found - no GPU attached to this container"

echo ""
echo "Logs if anything above looks wrong:"
echo "  ~/vault/local/ollama.log"
echo "  ~/vault/STEK2035-chatbot/backend.log"
echo "  ~/vault/STEK2035-Chatbot_FrontEnd/stek-chatbot/frontend.log"
echo ""
echo "Frontend should be reachable on whatever port/URL this platform exposes"
echo "for port 3000 (check the data-lab.site dashboard for the public URL)."
