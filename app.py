from flask import Flask, render_template, request

import math
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from scipy.stats import norm
import io
import base64
import sys
print(sys.executable)
import yfinance as yf

app = Flask(__name__)


def get_spot_price(ticker):
    print("TICKER", yf.Ticker(ticker))
    data = yf.Ticker(ticker).history(period="1y")
    print("data", data, "close data")
    if data.empty:
        return None
    return data["Close"].iloc[-1]

def estimate_volatility(ticker, period="1y"):
    data = yf.Ticker(ticker).history(period=period)
    if data.empty:
        return None
    data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))
    daily_vol = data["log_return"].std()
    return daily_vol * np.sqrt(252)



def estimate_risk_free_rate():
    return 0.0446  # Approximate 5%


def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def binomial_tree_price(S, K, T, r, sigma, option_type):
    N = 4
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    a = np.exp(r * dt)
    p = (a - d) / (u - d)

    stock_tree = [[S * (u ** j) * (d ** (i - j)) for j in range(i + 1)] for i in range(N + 1)]
    option_tree = [[0] * (i + 1) for i in range(N + 1)]
    early_exercise = [[False] * (i + 1) for i in range(N + 1)]

    # Terminal payoff
    for j in range(N + 1):
        if option_type == "call":
            option_tree[N][j] = max(0, stock_tree[N][j] - K)
        else:
            option_tree[N][j] = max(0, K - stock_tree[N][j])

    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            average = (p * option_tree[i + 1][j + 1] + (1 - p) * option_tree[i + 1][j])
            expected_value = np.exp(-r * dt) * average
            intrinsic = max(0, stock_tree[i][j] - K) if option_type == "call" else max(0, K - stock_tree[i][j])
            if intrinsic > expected_value:
                option_tree[i][j] = intrinsic
                early_exercise[i][j] = True
            else:
                option_tree[i][j] = expected_value

    # Build and draw the graph
    G = nx.DiGraph()
    pos = {}
    node_labels = {}
    node_colors = []

    for i in range(N + 1):
        for j in range(i + 1):
            node_id = (i, j)
            G.add_node(node_id)
            pos[node_id] = (i, 2 * j - N)
            if early_exercise[i][j]:
                    node_colors.append("red")  # Red = Early exercise
            else:
                node_colors.append("lightblue")  # Normal nodes are blue
            node_labels[node_id] = f"Stock={stock_tree[i][j]:.2f}\nOption={option_tree[i][j]:.2f}"
            
    for i in range(N):
        for j in range(i + 1):
            G.add_edge((i, j), (i + 1, j))      # down
            G.add_edge((i, j), (i + 1, j + 1))  # up
    nodes_sorted = sorted(G.nodes(), key=lambda x: (x[0], x[1]))
    node_colors = [G.nodes[n].get("color", "lightblue") for n in nodes_sorted]
    nx.draw(G, pos, nodelist=nodes_sorted, node_color=node_colors, node_size=2000)

    plt.figure(figsize=(12, 7))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, edgecolors='black')
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')
    
    plt.title("Binomial Tree")
    plt.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")


    return option_tree[0][0], image_base64
def calculate_greeks(S, K, T, r, sigma, option_type):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    theta = (
        - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) 
        - r * K * math.exp(-r * T) * norm.cdf(d2)
        if option_type == "call"
        else
        - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) 
        + r * K * math.exp(-r * T) * norm.cdf(-d2)
    )
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100  # scaled by 100
    rho = (
        K * T * math.exp(-r * T) * norm.cdf(d2) / 100
        if option_type == "call"
        else -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
    )

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,
        "Vega": vega,
        "Rho": rho
    }

def delta_black_scholes(S, K, T, r, sigma, option_type):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * math.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def gamma_black_scholes(S, K, T, r, sigma, option_type):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))

def theta_black_scholes(S, K, T, r, sigma, option_type):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    if option_type == "call":
        term1 = -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
        term2 = -r * K * math.exp(-r*T) * norm.cdf(d2)
        return term1 + term2
    else:
        term1 = -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
        term2 = r * K * math.exp(-r*T) * norm.cdf(-d2)
        return term1 + term2

def vega_black_scholes(S, K, T, r, sigma, option_type):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * math.sqrt(T))
    return S * norm.pdf(d1) * math.sqrt(T)

def rho_black_scholes(S, K, T, r, sigma, option_type):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return K * T * math.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * math.exp(-r * T) * norm.cdf(-d2)

def generate_greek_data(greek_func, S_min, S_max, K, T, r, sigma, option_type=None):
    S_values = np.linspace(S_min, S_max, 100)
    y_values = []
    for S in S_values:
        y = greek_func(S, K, T, r, sigma, option_type)
        y_values.append(y)
    return S_values, y_values

def plot_generic(x, y, x_label, y_label, title, vline=None):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if vline is not None:
        plt.axvline(x=vline, color='red', linestyle='--', label='Strike Price (K)')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64


    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(S_values, delta_vals, label="Delta")
    plt.plot(S_values, gamma_vals, label="Gamma")
    plt.plot(S_values, theta_vals, label="Theta")
    plt.plot(S_values, vega_vals, label="Vega")
    plt.plot(S_values, rho_vals, label="Rho")
    plt.xlabel("Underlying Price (S)")
    plt.ylabel("Greek Value")
    plt.title("Option Greeks vs Underlying Price")
    plt.legend()
    plt.grid(True)

    # Convert to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    greek_chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()

    return greek_chart_base64



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/price", methods=["POST"])
def price():
    input_mode = request.form.get("input_mode")
    if input_mode== "manual":
        # Get form data
        S = float(request.form["S"])
        K = float(request.form["K"])
        T = float(request.form["T"])
        r = float(request.form["r"])
        sigma = float(request.form["sigma"])
        option_type = request.form["option_type"]
    else:        
        ticker = request.form['ticker'].strip().upper().replace('$', '')
        K = float(request.form["K"])
        T = float(request.form["T"])
        option_type = request.form["option_type"]

        # Estimate missing values from ticker
        S = get_spot_price(ticker)
        sigma = estimate_volatility(ticker)

        if S is None or sigma is None:
            return "Failed to retrieve stock data. Please check the ticker symbol and try again.", 400

        r = estimate_risk_free_rate()    # placeholder
        

    # Calculate option price
    bs_price = black_scholes(S, K, T, r, sigma, option_type)
    tree_price, tree_visual = binomial_tree_price(S, K, T, r, sigma, option_type)
    
    delta = delta_black_scholes(S, K, T, r, sigma, option_type)
    gamma = gamma_black_scholes(S, K, T, r, sigma, option_type)
    theta = theta_black_scholes(S, K, T, r, sigma, option_type)
    vega = vega_black_scholes(S, K, T, r, sigma, option_type)
    rho = rho_black_scholes(S, K, T, r, sigma, option_type)
    greeks = {
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,
        "Vega": vega,
        "Rho": rho
    }
    greek_descriptions = {
    "Delta": "Rate of change of option price w.r.t. underlying price",
    "Gamma": "Rate of change of Delta w.r.t. underlying price",
    "Theta": "Time decay — change in option price over time",
    "Vega": "Sensitivity to volatility",
    "Rho": "Sensitivity to interest rate"
}
    S_min, S_max = K * 0.5, K * 1.5
    delta_x, delta_y = generate_greek_data(delta_black_scholes, S_min, S_max, K, T, r, sigma, option_type)
    gamma_x, gamma_y = generate_greek_data(gamma_black_scholes, S_min, S_max, K, T, r, sigma, option_type)
    theta_x, theta_y = generate_greek_data(theta_black_scholes, S_min, S_max, K, T, r, sigma, option_type)
    vega_x, vega_y = generate_greek_data(vega_black_scholes, S_min, S_max, K, T, r, sigma, option_type)
    rho_x, rho_y = generate_greek_data(rho_black_scholes, S_min, S_max, K, T, r, sigma, option_type)

    delta_plot = plot_generic(delta_x, delta_y, "Underlying Price (S)", "Delta", "Delta vs Underlying Price", vline=K)
    gamma_plot = plot_generic(gamma_x, gamma_y, "Underlying Price (S)", "Gamma", "Gamma vs Underlying Price", vline=K)
    theta_plot = plot_generic(theta_x, theta_y, "Underlying Price (S)", "Theta", "Theta vs Underlying Price", vline=K)
    vega_plot = plot_generic(vega_x, vega_y, "Underlying Price (S)", "Vega", "Vega vs Underlying Price", vline=K)
    rho_plot = plot_generic(rho_x, rho_y, "Underlying Price (S)", "Rho", "Rho vs Underlying Price", vline=K)
    #greek_chart = generate_greek_graphs(S, K, T, r, sigma, option_type)
    # Render a result page with the price and tree image
    return render_template("results.html", price1=bs_price, price2=tree_price, option_type=option_type, tree_image=tree_visual, greeks=greeks ,greek_descriptions=greek_descriptions,delta_plot=delta_plot,
        gamma_plot=gamma_plot,
        theta_plot=theta_plot,
        vega_plot=vega_plot,
        rho_plot=rho_plot,
        S=S,
        K=K,
        sigma=sigma,
        T=T,
        r=r*100)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render will set the PORT env variable
    app.run(host="0.0.0.0", port=port)
