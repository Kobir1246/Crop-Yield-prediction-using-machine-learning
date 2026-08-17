import io, os, base64
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request, Response

app = Flask(__name__)

def _clean_name(name: str) -> str:
    for prefix in ["Crop Name_", "District_", "Season_"]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

app.jinja_env.filters["clean_name"] = _clean_name

BASE = os.path.dirname(os.path.abspath(__file__))

model_xgb = joblib.load(os.path.join(BASE, "xgb_final_model.pkl"))
ref = joblib.load(os.path.join(BASE, "xgb_final_ref.pkl"))
xgb_feature_names = ref["feature_names"]
XGB_MODEL_LABEL = "Tuned XGBoost Regressor (with AP Ratio)"

model_stack = joblib.load(os.path.join(BASE, "stack_final_model.pkl"))
stack_feature_names = list(model_stack.feature_names_in_)
STACK_MODEL_LABEL = "Advanced Stacking Ensemble (RF + GBR + XGBoost + Ridge)"

ALL_DISTRICTS = ref["ALL_DISTRICTS"]
ALL_SEASONS = ref["ALL_SEASONS"]
ALL_CROPS = ref["ALL_CROPS"]

MODEL_CHOICES = [
    {"id": "xgb",   "label": XGB_MODEL_LABEL},
    {"id": "stack", "label": STACK_MODEL_LABEL},
]

XGB_RMSE = 2.45
STACK_RMSE = 3.26

_df = pd.read_csv(os.path.join(BASE, "data", "SPAS_with_months.csv"))
_df = _df[_df["Area"] > 0].copy()
_df = _df[_df["Production"] > 0].copy()
_df["Season"] = _df["Season"].fillna(_df["Season"].mode()[0])
_df["ap_ratio"] = _df["Production"] / _df["Area"]

_crop_defaults = {}
for crop in ALL_CROPS:
    cd = _df[_df["Crop Name"] == crop]
    if len(cd) > 0:
        _crop_defaults[crop] = {
            "district": cd["District"].mode()[0],
            "season": cd["Season"].mode()[0],
            "area": float(cd["Area"].median()),
            "avg_temp": float(cd["Avg Temp"].median()),
            "avg_humidity": float(cd["Avg Humidity"].median()),
            "max_temp": float(cd["Max Temp"].median()),
            "min_temp": float(cd["Min Temp"].median()),
            "max_rh": float(cd["Max Relative Humidity"].median()),
            "min_rh": float(cd["Min Relative Humidity"].median()),
            "rainfall": float(cd["Rainfall_mm"].median()),
            "ap_ratio": float(cd["ap_ratio"].median()),
        }


def validate_input(data: dict) -> dict | None:
    errors = {}
    required = ["crop", "district", "season", "area", "avg_temp", "avg_humidity",
                "max_temp", "min_temp", "max_rh", "min_rh", "rainfall"]
    for field in required:
        if not data.get(field, "").strip():
            errors[field] = "This field is required"

    try:
        area = float(data.get("area", 0))
        if area <= 0:
            errors["area"] = "Area must be greater than 0"
    except ValueError:
        errors["area"] = "Invalid number"

    for fld, fname in [("avg_temp", "Avg Temp"), ("avg_humidity", "Avg Humidity"),
                        ("max_temp", "Max Temp"), ("min_temp", "Min Temp"),
                        ("max_rh", "Max RH"), ("min_rh", "Min RH"),
                        ("rainfall", "Rainfall")]:
        try:
            v = float(data.get(fld, 0))
        except ValueError:
            errors[fld] = f"{fname}: invalid number"

    if not errors:
        max_t = float(data["max_temp"])
        min_t = float(data["min_temp"])
        if min_t > max_t:
            errors["min_temp"] = "Min temp cannot exceed Max temp"
            errors["max_temp"] = "Max temp cannot be below Min temp"

        for fld, v, lo, hi, nm in [
            ("avg_humidity", float(data["avg_humidity"]), 0, 100, "Avg Humidity"),
            ("max_rh", float(data["max_rh"]), 0, 100, "Max RH"),
            ("min_rh", float(data["min_rh"]), 0, 100, "Min RH"),
        ]:
            if not (lo <= v <= hi):
                errors[fld] = f"{nm} must be between {lo} and {hi}"

        if float(data["avg_humidity"]) < float(data["min_rh"]):
            errors["avg_humidity"] = "Avg Humidity seems low relative to Min RH"
        if float(data["max_rh"]) < float(data["min_rh"]):
            errors["max_rh"] = "Max RH cannot be below Min RH"

        if float(data["rainfall"]) < 0:
            errors["rainfall"] = "Rainfall cannot be negative"
        if float(data["area"]) > 1_000_000:
            errors["area"] = "Area seems unusually large (>1M ha)"

    return errors if errors else None


def _build_row(data: dict, fn_list: list) -> dict:
    area = float(data.get("area", 0))
    avg_temp = float(data.get("avg_temp", 0))
    avg_humidity = float(data.get("avg_humidity", 0))
    max_temp = float(data.get("max_temp", 0))
    min_temp = float(data.get("min_temp", 0))
    max_rh = float(data.get("max_rh", 0))
    min_rh = float(data.get("min_rh", 0))
    rainfall = float(data.get("rainfall", 0))

    row = {f: 0.0 for f in fn_list}
    row["Area"] = area
    row["Avg Temp"] = avg_temp
    row["Avg Humidity"] = avg_humidity
    row["Max Temp"] = max_temp
    row["Min Temp"] = min_temp
    row["Max Relative Humidity"] = max_rh
    row["Min Relative Humidity"] = min_rh
    row["Rainfall_mm"] = rainfall
    row["Temp_Range"] = max_temp - min_temp
    row["Humidity_Range"] = max_rh - min_rh
    row["Climate_Index"] = rainfall * avg_temp

    if "AP Ratio" in fn_list:
        crop = data.get("crop", "")
        defaults = _crop_defaults.get(crop, {})
        row["AP Ratio"] = defaults.get("ap_ratio", float(_df["ap_ratio"].median()))

    for col_key, prefix in [("district", "District_"), ("season", "Season_"), ("crop", "Crop Name_")]:
        val = data.get(col_key, "")
        col = f"{prefix}{val}"
        if col in fn_list:
            row[col] = 1.0

    return row


def preprocess_xgb(data: dict) -> pd.DataFrame:
    row = _build_row(data, xgb_feature_names)
    return pd.DataFrame([row])[xgb_feature_names]


def preprocess_stack(data: dict) -> pd.DataFrame:
    row = _build_row(data, stack_feature_names)
    return pd.DataFrame([row])[stack_feature_names]


SHAP_OK = False
_explainer_xgb = None
_explainer_stack = None
_xgb_in_stack = None

try:
    import shap
    import matplotlib
    matplotlib.use("Agg")
    _explainer_xgb = shap.TreeExplainer(model_xgb)
    _xgb_in_stack = model_stack.estimators_[2]
    _explainer_stack = shap.TreeExplainer(_xgb_in_stack)
    SHAP_OK = True
except Exception:
    pass


def compute_shap(feature_vector: pd.DataFrame, model_type: str) -> list:
    if not SHAP_OK:
        return []
    if model_type == "stack":
        explainer = _explainer_stack
        fn = stack_feature_names
    else:
        explainer = _explainer_xgb
        fn = xgb_feature_names
    shap_values = explainer.shap_values(feature_vector)
    vals = shap_values[0]
    out = []
    for name, val in zip(fn, vals):
        out.append({"name": name, "value": round(float(val), 4), "abs_value": abs(val)})
    out.sort(key=lambda x: x["abs_value"], reverse=True)
    return out


def shap_beeswarm_b64(feature_vector: pd.DataFrame, model_type: str) -> str:
    if not SHAP_OK:
        return ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if model_type == "stack":
            explainer = _explainer_stack
        else:
            explainer = _explainer_xgb
        shap_values = explainer.shap_values(feature_vector)
        fig, _ = plt.subplots(figsize=(5, 3.2))
        shap.summary_plot(shap_values, feature_vector, plot_type="bar", show=False,
                          max_display=8, color="#2e7d32", axis_color="#333")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    result = None
    error = None
    field_errors = None
    shap_data = None
    shap_img = None
    selected_model = "xgb"
    form_data = {}

    if request.method == "POST":
        data = request.form.to_dict()
        form_data = data
        selected_model = data.get("model_choice", "xgb")

        field_errors = validate_input(data)

        if not field_errors:
            try:
                if selected_model == "stack":
                    X = preprocess_stack(data)
                    pred = float(model_stack.predict(X)[0])
                    model_label = STACK_MODEL_LABEL
                else:
                    X = preprocess_xgb(data)
                    pred = float(model_xgb.predict(X)[0])
                    model_label = XGB_MODEL_LABEL

                pred = max(pred, 0.0)

                rmse = STACK_RMSE if selected_model == "stack" else XGB_RMSE
                ci_lo = max(0.0, round(pred - 1.96 * rmse, 2))
                ci_hi = round(pred + 1.96 * rmse, 2)

                result = {
                    "yield": round(pred, 2),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "crop": data.get("crop", ""),
                    "district": data.get("district", ""),
                    "season": data.get("season", ""),
                    "area": data.get("area", ""),
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "model": model_label,
                }

                shap_data = compute_shap(X, selected_model)
                shap_img = shap_beeswarm_b64(X, selected_model)

            except Exception as e:
                error = str(e)

    return render_template(
        "index.html",
        districts=ALL_DISTRICTS,
        seasons=ALL_SEASONS,
        crops=ALL_CROPS,
        defaults=_crop_defaults,
        model_choices=MODEL_CHOICES,
        selected_model=selected_model,
        result=result,
        error=error,
        field_errors=field_errors,
        form_data=form_data,
        shap_data=shap_data,
        shap_img=shap_img,
        XGB_RMSE=XGB_RMSE,
        STACK_RMSE=STACK_RMSE,
    )


@app.route("/download")
def download():
    crop = request.args.get("crop", "")
    district = request.args.get("district", "")
    season = request.args.get("season", "")
    area = request.args.get("area", "")
    yld = request.args.get("yield", "")
    ci_lo = request.args.get("ci_lo", "")
    ci_hi = request.args.get("ci_hi", "")
    model_used = request.args.get("model", "")
    date = request.args.get("date", "")

    csv = (
        f"Crop,District,Season,Area (ha),Yield (t/ha),95% CI Lower,95% CI Upper,Model,Prediction Date\n"
        f"{crop},{district},{season},{area},{yld},{ci_lo},{ci_hi},{model_used},{date}\n"
    )
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=prediction_result.csv"},
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=5000)
    except ImportError:
        app.run(debug=False, host="0.0.0.0", port=5000)
