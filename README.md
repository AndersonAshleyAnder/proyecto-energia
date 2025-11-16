---

### ✅ **2. Crear `README.md`**
Este archivo describe tu proyecto y cómo usarlo.

**Contenido sugerido:**
```markdown
# Proyecto Energía ⚡
Predicción del precio de energía en la Bolsa Nacional usando Dash y Machine Learning.

## 🚀 Instalación
```bash
git clone https://github.com/AndersonAshleyAnder/proyecto-energia.git
cd proyecto-energia
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Si sale error del comando anterior ejecutar primero este
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install scikit-learn
python -m pip install dash plotly pandas
