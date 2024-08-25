import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import collections
import random
import sys
import warnings
warnings.filterwarnings('ignore')

# Price Simulation Class
class PriceSimulator:
    def __init__(self, initial_price=100, mu=0.0001, sigma=0.2, days=22, minutes_per_day=480, dt=1, seed=4):
        self.initial_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.days = days
        self.minutes_per_day = minutes_per_day
        self.T = self.days * self.minutes_per_day
        self.dt = dt
        self.seed = seed
        self.N = int(self.T / self.dt)
        self.brownian_motion_prices = np.zeros(self.N)
        self.price_df = pd.DataFrame()

    def simulate_brownian_motion_prices(self):
        np.random.seed(self.seed)
        self.brownian_motion_prices[0] = round(self.initial_price, 2)
        
        for t in range(1, self.N):
            Z_t = np.random.randn()
            self.brownian_motion_prices[t] = round(self.brownian_motion_prices[t-1] + \
                                            self.sigma * np.sqrt(self.dt) * Z_t, 2)
        
        return self.brownian_motion_prices
  
    def generate_trading_days(self):
        self.start_date = datetime(2023, 1, 1)
        self.trading_days = []
        while len(self.trading_days) < self.days:
            if self.start_date.weekday() < 5:
                self.trading_days.append(self.start_date)
            self.start_date += timedelta(days=1)

    def generate_time_series(self):
        self.time_series = []
        for day in self.trading_days:
            for minute in range(self.minutes_per_day):
                self.time_series.append(day + timedelta(minutes=minute))

    def create_dataframe(self, prices):
        self.generate_trading_days()
        self.generate_time_series()

        if len(prices) != len(self.time_series):
            raise ValueError(f"Mismatch in lengths: Prices({len(prices)}) vs Time Series({len(self.time_series)})")

        self.price_df = pd.DataFrame({
            'datetime': self.time_series,
            'price': prices
        })

        self.price_df['date'] = self.price_df['datetime'].dt.strftime('%Y-%m-%d')
        self.price_df['time'] = self.price_df['datetime'].dt.time
        self.price_df.drop(columns=['datetime'], inplace=True)

        self.price_df['tick_by_tick_return'] = self.price_df['price'].pct_change().fillna(0)
        self.price_df = self.price_df[["date", "time", "price", "tick_by_tick_return"]]
        self.price_df.to_csv("data/pricing_data.csv", index=False)
        return self.price_df
    
    def plot_prices(self, figsize:tuple=(15, 5)):
        plt.figure(figsize=figsize)
        plt.plot(self.price_df["price"], label="Asset Price")
        plt.xlabel("Ticks")
        plt.ylabel("Price")
        plt.title("Asset Price")
        plt.legend()
        plt.grid(visible=True)
        plt.show()

# Agent Classes
class Agent:
    def __init__(self, id, initial_cash, initial_inventory, aggressiveness):
        self.id = id
        self.cash = initial_cash
        self.inventory = initial_inventory
        self.aggressiveness = aggressiveness
        self.order_history = []
        self.is_liquidated = False

class GamblerAgent(Agent):
    def __init__(self, id, initial_cash, initial_inventory, aggressiveness):
        super().__init__(id, initial_cash, initial_inventory, aggressiveness)
        self.momentum_period = random.choice([1, 2, 3, 4, 5])  # Choose a random momentum period between 1 and 5 days
        self.momentum_tag = f"{self.momentum_period}-day Momentum"
        self.current_position = None  # Track current open position
        self.hold_time = None  # To store the holding period
        self.last_trade_time = None  # To track when the position was opened

        # DataFrame to track orders, cash, PnL, etc.
        self.metrics_df = pd.DataFrame(columns=[
            "momentum_trader", "aggressiveness", "timestamp", "cash", 
            "inventory", "pnl", "order_type", "order_price", "order_size",
            "order_dollar_value", "position"
        ])
        self.pnl = 0  # Initialize PnL

    def place_order(self, recent_prices, current_time):
        if self.is_liquidated:
            return None

        # Check if it's time to close a position
        if self.current_position and current_time >= self.last_trade_time + self.hold_time:
            order_type = "sell" if self.current_position['type'] == "buy" else "buy"
            price = round(recent_prices[-1], 2)
            size = round(self.current_position['size'], 2)
            return (order_type, price, size)

        # Open a new position if no current position
        if not self.current_position:
            # Determine how many days of data to use based on the selected momentum period
            period_length = 480 * self.momentum_period  # 480 ticks per day
            # Calculate momentum based on the selected period
            if len(recent_prices) < period_length:
                period_length = len(recent_prices)  # Ensure we don't exceed available data length
            momentum = np.sum(np.diff(recent_prices[-period_length:]))
            # Determine direction based on momentum
            direction = 1 if momentum > 0 else -1
            order_type = "buy" if direction > 0 else "sell"
            # Calculate order size and price
            price = round(recent_prices[-1] * (1 + (0.01 * self.aggressiveness) * direction), 2)
            order_size = round((self.cash * self.aggressiveness) / price, 2) / 10
            # Ensure the agent can afford the position
            if self.cash < (abs(order_size) * price):
                return None  # Skip the trade if not enough cash
            return (order_type, price, abs(order_size))

        return None

    def calculate_pnl(self, order_type, price, size):
        """Calculate PnL based on the current position and the trade being executed."""
        if self.current_position is None:
            trade_pnl = 0
        elif order_type == "sell":  # PnL only at closing a trade
            # Selling a position, PnL = (Current Price - Buy Price) * Size
            trade_pnl = (price - self.current_position['price']) * size
        elif order_type == "buy":  # PnL only at closing a trade
            # Buying to cover a sell position, PnL = (Sell Price - Current Price) * Size
            trade_pnl = (self.current_position['price'] - price) * size

        return trade_pnl

    def order_executed(self, trade_price, trade_size, trade_type, current_time):

        # print(f"{trade_type} Order for Gambler Agent {agent.id} Executed")

        if trade_type == "buy":
            if self.current_position:
                # print(f"Current Position: {self.current_position['type']}, Price: {self.current_position['price']}, Size: {self.current_position['size']}")
                # print("Closing Existing SELL Position by Buying")
                self.inventory += trade_size
                self.cash -= trade_price * trade_size
                trade_pnl = self.calculate_pnl(trade_type, price, trade_size)
                self.pnl += trade_pnl  # Update cumulative PnL
                self.record_metrics(current_time, price, trade_size, trade_type, trade_pnl, "close")
                self.order_history.append((price, trade_size))
                self.current_position = None  # Reset current position after closing
            else:
                # print("Opening New BUY Position")
                # Set a random holding period between 240 (half a day) and 4800 (10 days) ticks
                self.hold_time = random.randint(30, 240)
                self.last_trade_time = current_time  # Set the time when the position was opened
                self.inventory += trade_size
                self.cash -= trade_price * trade_size
                trade_pnl = self.calculate_pnl(trade_type, price, trade_size)
                self.pnl += trade_pnl  # Update cumulative PnL
                self.record_metrics(current_time, price, trade_size, trade_type, trade_pnl, "open")
                self.order_history.append((price, trade_size))
                self.current_position = {"type": trade_type, "price": price, "size": abs(trade_size)}

        elif trade_type == "sell":
            if self.current_position:
                # print(f"Current Position: {self.current_position['type']}, Price: {self.current_position['price']}, Size: {self.current_position['size']}")
                # print("Closing Existing BUY Position by Selling")
                self.inventory -= trade_size
                self.cash += trade_price * trade_size
                trade_pnl = self.calculate_pnl(trade_type, price, trade_size)
                self.pnl += trade_pnl  # Update cumulative PnL
                self.record_metrics(current_time, price, trade_size, trade_type, trade_pnl, "close")
                self.order_history.append((price, trade_size))
                self.current_position = None  # Reset current position after closing
            else:
                # print("Opening New SELL Position")
                # Set a random holding period between 240 (half a day) and 4800 (10 days) ticks
                self.hold_time = random.randint(30, 240)
                self.last_trade_time = current_time  # Set the time when the position was opened
                self.inventory -= trade_size
                self.cash += trade_price * trade_size
                trade_pnl = self.calculate_pnl(trade_type, price, trade_size)
                self.pnl += trade_pnl  # Update cumulative PnL
                self.record_metrics(current_time, price, trade_size, trade_type, trade_pnl, "open")
                self.order_history.append((price, trade_size))
                self.current_position = {"type": trade_type, "price": price, "size": abs(trade_size)}

        if self.cash < 0:
            self.is_liquidated = True  # Liquidate if cash is negative
            # self.record_metrics_for_liquidation()  # Update metrics before exporting
            # self.export_metrics()  # Export metrics to CSV upon liquidation
        
        # print(f"Order Type: {trade_type}, Price: {price}, Size: {trade_size}")
        # print(f"Trade PnL: {trade_pnl}")
        # print(f"Cumulative PnL: {self.pnl}")
        # print("Metrics DataFrame")
        # print(self.metrics_df)
        # pause_and_resume()

    def record_metrics(self, timestamp, price, size, order_type, trade_pnl, position):
        """Record relevant metrics in the DataFrame."""
        new_entry = pd.DataFrame({
            "momentum_trader": self.momentum_tag,
            "aggressiveness": self.aggressiveness,
            "timestamp": [timestamp],
            "cash": [self.cash],
            "inventory": [self.inventory],
            "pnl": [trade_pnl],
            "order_type": [order_type],
            "order_price": [price],
            "order_size": [size],
            "order_dollar_value":[price * size],
            "position":position
        })
        self.metrics_df = pd.concat([self.metrics_df, new_entry], ignore_index=True)

    def record_metrics_for_liquidation(self):
        """Update the last entry in the DataFrame to reflect the negative cash balance."""
        if not self.metrics_df.empty:
            self.metrics_df.loc[self.metrics_df.index[-1], "cash"] = self.cash

    def export_metrics(self):
        """Export the metrics DataFrame to a CSV file."""
        file_name = f"gambler_{self.id}_metrics.csv"
        self.metrics_df.to_csv(f"data/gamblers/{file_name}", index=False)
        print(f"Metrics for Gambler {self.id} exported to {file_name}.")

class HedgeFundAgent(Agent):
    def __init__(self, id, initial_cash, initial_inventory, aggressiveness):
        super().__init__(id, initial_cash, initial_inventory, aggressiveness)
        self.hold_time = 100  # Hedge fund holds positions for 100 ticks
        self.current_position = None  # Track current open position
        self.last_trade_time = None  # To track when the position was opened

        # DataFrame to track orders, cash, PnL, etc.
        self.metrics_df = pd.DataFrame(columns=[
            "timestamp", "cash", "inventory", "pnl", "order_type", "order_price", "order_size", "position"
        ])
        self.pnl = 0  # Initialize PnL

    def place_order(self, current_price, future_price, current_time):
        if self.is_liquidated:
            return None

        # Check if it's time to close a position
        if self.current_position and current_time >= self.last_trade_time + self.hold_time:
            order_type = "sell" if self.current_position['type'] == "buy" else "buy"
            price = round(current_price, 2)
            size = round(self.current_position['size'], 2)
            return (order_type, price, size)

        # Open a new position if no current position
        if not self.current_position:
            # Decide to buy or sell based on future price
            if future_price > current_price:
                order_type = "buy"
            else:
                order_type = "sell"

            # Calculate order size and price
            price = round(current_price, 2)
            order_size = round((self.cash * self.aggressiveness) / price, 2)

            # Ensure the agent can afford the position
            if self.cash < (abs(order_size) * price):
                return None  # Skip the trade if not enough cash

            return (order_type, price, abs(order_size))

        return None

    def calculate_pnl(self, order_type, price, size):
        """Calculate PnL based on the current position and the trade being executed."""
        if self.current_position is None:
            trade_pnl = 0
        elif order_type == "sell":  # PnL only at closing a trade
            trade_pnl = (price - self.current_position['price']) * size
        elif order_type == "buy":  # PnL only at closing a trade
            trade_pnl = (self.current_position['price'] - price) * size

        return trade_pnl

    def order_executed(self, trade_price, trade_size, trade_type, current_time):
        if trade_type == "buy":
            if self.current_position:
                # Closing an existing position
                self.inventory += trade_size
                self.cash -= trade_price * trade_size
                trade_pnl = self.calculate_pnl(trade_type, trade_price, trade_size)
                self.pnl += trade_pnl
                self.record_metrics(current_time, trade_price, trade_size, trade_type, trade_pnl, "close")
                self.current_position = None  # Reset current position after closing
            else:
                # Opening a new position
                self.inventory += trade_size
                self.cash -= trade_price * trade_size
                self.record_metrics(current_time, trade_price, trade_size, trade_type, 0, "open")
                self.current_position = {"type": trade_type, "price": trade_price, "size": abs(trade_size)}
                self.last_trade_time = current_time  # Set the time when the position was opened
        elif trade_type == "sell":
            if self.current_position:
                # Closing an existing position
                self.inventory -= trade_size
                self.cash += trade_price * trade_size
                trade_pnl = self.calculate_pnl(trade_type, trade_price, trade_size)
                self.pnl += trade_pnl
                self.record_metrics(current_time, trade_price, trade_size, trade_type, trade_pnl, "close")
                self.current_position = None  # Reset current position after closing

            else:
                # Opening a new position
                self.inventory -= trade_size
                self.cash += trade_price * trade_size
                self.record_metrics(current_time, trade_price, trade_size, trade_type, 0, "open")
                self.current_position = {"type": trade_type, "price": trade_price, "size": abs(trade_size)}
                self.last_trade_time = current_time  # Set the time when the position was opened

        if self.cash < 0:
            self.is_liquidated = True  # Liquidate if cash is negative
            self.export_metrics()  # Export metrics to CSV upon liquidation

    def record_metrics(self, timestamp, price, size, order_type, trade_pnl, position):
        """Record relevant metrics in the DataFrame."""
        new_entry = pd.DataFrame({
            "timestamp": [timestamp],
            "cash": [self.cash],
            "inventory": [self.inventory],
            "pnl": [trade_pnl],
            "order_type": [order_type],
            "order_price": [price],
            "order_size": [size],
            "position": [position]
        })
        self.metrics_df = pd.concat([self.metrics_df, new_entry], ignore_index=True)

    def export_metrics(self):
        """Export the metrics DataFrame to a CSV file."""
        file_name = f"hedge_fund_{self.id}_metrics.csv"
        self.metrics_df.to_csv(f"data/hedge_funds/{file_name}", index=False)
        print(f"Metrics for Hedge Fund {self.id} exported to {file_name}.")

class MarketMaker(Agent):
    def __init__(self, id, initial_cash, initial_inventory, aggressiveness=0.1, risk_aversion=0.1):
        super().__init__(id, initial_cash, initial_inventory, aggressiveness)
        self.risk_aversion = risk_aversion
        self.spread_history = []  # Track bid-ask spread over time
        self.volume_history = []  # Track quoted volumes over time
        self.inventory_history = []  # Track inventory over time
        self.pnl_history = []  # Track PnL over time
        self.time_history = []  # Track timestamps for analysis

        # DataFrame to track orders, cash, PnL, etc.
        self.metrics_df = pd.DataFrame(columns=[
            "timestamp", "cash", "inventory", "pnl", "order_type", "order_price", "order_size", "position"
        ])

    def order_executed(self, trade_price, trade_size, trade_type, current_time):
        if trade_type == "buy":
            self.inventory += trade_size
            self.cash -= trade_price * trade_size
        elif trade_type == "sell":
            self.inventory -= trade_size
            self.cash += trade_price * trade_size

        # Record PnL after each trade
        pnl = self.cash + (self.inventory * trade_price)
        self.pnl_history.append(pnl)
        position = "close" if self.inventory == 0 else "open"
        self.record_metrics(current_time, trade_price, trade_size, trade_type, pnl, position)

        if self.cash < 0:
            self.is_liquidated = True  # Liquidate if cash is negative
            self.export_metrics()
            print("MARKET MAKER HAS GONE BANKRUPT")
            pause_and_resume()

    def record_metrics(self, timestamp, price, size, order_type, trade_pnl, position):
        """Record relevant metrics in the DataFrame."""
        new_entry = pd.DataFrame({
            "timestamp": [timestamp],
            "cash": [self.cash],
            "inventory": [self.inventory],
            "pnl": [trade_pnl],
            "order_type": [order_type],
            "order_price": [price],
            "order_size": [size],
            "position": [position]
        })
        self.metrics_df = pd.concat([self.metrics_df, new_entry], ignore_index=True)

    def export_metrics(self):
        """Export the metrics DataFrame to a CSV file."""
        file_name = f"market_maker_{self.id}_metrics.csv"
        self.metrics_df.to_csv(f"data/market_makers/{file_name}", index=False)
        print(f"Metrics for Market Maker {self.id} exported to {file_name}.")

class MarketMaker(Agent):
    def __init__(self, id, initial_cash, initial_inventory, aggressiveness=0.1, risk_aversion=0.05):
        super().__init__(id, initial_cash, initial_inventory, aggressiveness)
        self.risk_aversion = risk_aversion
        self.spread_history = []  # Track bid-ask spread over time
        self.volume_history = []  # Track quoted volumes over time
        self.inventory_history = []  # Track inventory over time
        self.pnl_history = []  # Track PnL over time
        self.time_history = []  # Track timestamps for analysis

        # DataFrame to track orders, cash, PnL, etc.
        self.metrics_df = pd.DataFrame(columns=[
            "timestamp", "cash", "inventory", "pnl", "order_type", "order_price", "order_size", "position"
        ])

    def calculate_optimal_prices(self, current_price, inventory):
        # Modify the risk aversion impact on price calculations
        lambda_b = max(0, 1 - 0.5 * self.risk_aversion * inventory)  # Reducing the impact of risk aversion
        lambda_a = max(0, 1 + 0.5 * self.risk_aversion * inventory)  # Reducing the impact of risk aversion
        
        bid_price = current_price - (0.5 / self.risk_aversion) * np.log(1 + self.risk_aversion * lambda_b)
        ask_price = current_price + (0.5 / self.risk_aversion) * np.log(1 + self.risk_aversion * lambda_a)
        
        self.order_history.append((bid_price, ask_price))
        self.spread_history.append(ask_price - bid_price)  # Record the bid-ask spread
        
        return (bid_price, ask_price)

    def calculate_optimal_volume(self, current_price, order_book, inventory):
        # Factor 1: Adjust based on inventory
        inventory_factor = max(0.5, 1 - 0.5 * self.risk_aversion * abs(inventory))  # Reduce impact, increase volume
        
        # Extract all buy and sell volumes
        buy_volumes = [order["size"] for orders_at_price in order_book.buy_orders.values() for order in orders_at_price]
        sell_volumes = [order["size"] for orders_at_price in order_book.sell_orders.values() for order in orders_at_price]

        # Combine buy and sell volumes to get the overall depth
        all_volumes = buy_volumes + sell_volumes
        
        if len(all_volumes) > 0:
            depth_factor = min(2, np.mean(all_volumes) / 5000)  # Increase volume with higher market depth
        else:
            depth_factor = 0.5  # Base depth factor

        # Factor 3: Adjust based on wealth and risk aversion
        wealth_factor = max(0.5, self.cash / (self.cash + inventory * current_price)) * (1 - self.risk_aversion)

        # Final optimal volume calculation
        optimal_volume = inventory_factor * depth_factor * wealth_factor * 5000  # Scale by a higher factor, e.g., 5000
        
        return max(10, round(optimal_volume, 2))  # Ensure the volume is at least 10
    def place_order(self, current_price, order_book, inventory):
        bid_price, ask_price = self.calculate_optimal_prices(current_price, inventory)
        bid_volume = self.calculate_optimal_volume(bid_price, order_book, inventory)
        ask_volume = self.calculate_optimal_volume(ask_price, order_book, inventory)
        
        # Store the order in the order history
        self.order_history.append(("buy", bid_price, bid_volume))
        self.order_history.append(("sell", ask_price, ask_volume))
        self.volume_history.append((bid_volume, ask_volume))  # Record quoted volumes
        self.inventory_history.append(inventory)  # Record current inventory
        
        return bid_price, bid_volume, ask_price, ask_volume

    def analyze_spread_vs_inventory(self):
        """Plot the bid-ask spread as a function of inventory."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.inventory_history, self.spread_history, 'o-', color='purple')
        plt.xlabel('Inventory')
        plt.ylabel('Bid-Ask Spread')
        plt.title('Bid-Ask Spread vs Inventory')
        plt.grid(True)
        plt.savefig("data/spread_vs_inventory.png", dpi=300)

    def analyze_volume_vs_inventory(self):
        """Plot the quoted volumes as a function of inventory."""
        bid_volumes, ask_volumes = zip(*self.volume_history)
        plt.figure(figsize=(10, 6))
        plt.plot(self.inventory_history, bid_volumes, 'o-', color='blue', label='Bid Volume')
        plt.plot(self.inventory_history, ask_volumes, 'o-', color='orange', label='Ask Volume')
        plt.xlabel('Inventory')
        plt.ylabel('Volume')
        plt.title('Quoted Volumes vs Inventory')
        plt.legend()
        plt.grid(True)
        plt.savefig("data/volume_vs_inventory.png", dpi=300)

    def analyze_pnl_vs_inventory(self):
        """Plot PnL and Inventory as a function of time on two subplots."""
        fig, ax1 = plt.subplots(2, 1, figsize=(10, 12))

        # Subplot 1: PnL vs Inventory
        ax1[0].plot(range(len(self.pnl_history)), self.pnl_history, 'o-', color='green')
        ax1[0].set_xlabel('Inventory')
        ax1[0].set_ylabel('PnL')
        ax1[0].set_title('PnL vs Inventory')
        ax1[0].grid(True)

        # Subplot 2: Inventory over time
        ax1[1].plot(range(len(self.inventory_history)), self.inventory_history, 'o-', color='blue')
        ax1[1].set_xlabel('Time')
        ax1[1].set_ylabel('Inventory')
        ax1[1].set_title('Inventory Over Time')
        ax1[1].grid(True)

        plt.tight_layout()  # Adjust subplots to fit in the figure area.
        plt.savefig("data/pnl_vs_inventory.png", dpi=300)

    def analyze_pnl_vs_time(self):
        """Plot PnL over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.pnl_history)), self.pnl_history, color='red')
        plt.xlabel('Time')
        plt.ylabel('PnL')
        plt.title('PnL Over Time')
        plt.grid(True)
        plt.savefig("data/PnL.png", dpi=300)

    def analyze_inventory_vs_time(self):
        """Plot inventory over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.inventory_history)), self.inventory_history, color='blue')
        plt.xlabel('Time')
        plt.ylabel('Inventory')
        plt.title('Inventory Over Time')
        plt.grid(True)
        plt.savefig("data/inventory_over_time.png", dpi=300)

    def run_all_analyses(self):
        """Run all analyses to visualize the Market Maker's performance."""
        self.analyze_spread_vs_inventory()
        self.analyze_volume_vs_inventory()
        # self.analyze_pnl_vs_inventory()
        self.analyze_pnl_vs_time()
        self.analyze_inventory_vs_time()

# Order Book Class
class OrderBook:
    def __init__(self):
        self.buy_orders = collections.defaultdict(list)  # Dictionary to store buy orders
        self.sell_orders = collections.defaultdict(list)  # Dictionary to store sell orders
        self.trade_ledger = []  # List to record executed trades

    def add_order(self, agent_id, order_type, price, size, timestamp):
        order = {"agent_id": agent_id, "price": price, "size": size, "timestamp": timestamp}

        if order_type == "buy":
            self.buy_orders[price].append(order)
        elif order_type == "sell":
            self.sell_orders[price].append(order)

    def execute_trades(self, agents, current_price, timestamp):
        executed_trades = []

        # Sort buy and sell orders by price (highest buy first, lowest sell first)
        buy_prices = sorted(self.buy_orders.keys(), reverse=True)
        sell_prices = sorted(self.sell_orders.keys())

        # Execute trades where buy price >= sell price
        for buy_price in buy_prices:
            if buy_price < current_price:
                continue
            for sell_price in sell_prices:
                if sell_price > current_price or buy_price < sell_price:
                    continue

                buy_orders = self.buy_orders[buy_price]
                sell_orders = self.sell_orders[sell_price]

                # Sort by timestamp, then by volume for priority execution
                buy_orders.sort(key=lambda x: (x["timestamp"], -x["size"]))
                sell_orders.sort(key=lambda x: (x["timestamp"], -x["size"]))

                while buy_orders and sell_orders:
                    buy_order = buy_orders[0]
                    sell_order = sell_orders[0]

                    # Determine the trade size
                    trade_size = min(buy_order["size"], sell_order["size"])

                    # Execute trade
                    self.trade_ledger.append({
                        "timestamp": timestamp,
                        "buyer_id": buy_order["agent_id"],
                        "seller_id": sell_order["agent_id"],
                        "price": sell_price,
                        "size": trade_size
                    })
                    executed_trades.append(self.trade_ledger[-1])

                    # Update order sizes
                    buy_order["size"] -= trade_size
                    sell_order["size"] -= trade_size

                    # Order Executed - update each agent's ledger and metrics
                    if buy_order["agent_id"] == market_maker.id:
                        market_maker.order_executed(sell_price, trade_size, "buy", timestamp)
                    else:
                        agents[buy_order["agent_id"] - 1].order_executed(sell_price, trade_size, "buy", timestamp)

                    if sell_order["agent_id"] == market_maker.id:
                        market_maker.order_executed(sell_price, trade_size, "sell", timestamp)
                    else:
                        agents[sell_order["agent_id"] - 1].order_executed(sell_price, trade_size, "sell", timestamp)


                    # Remove orders if completely filled
                    if buy_order["size"] == 0:
                        buy_orders.pop(0)
                    if sell_order["size"] == 0:
                        sell_orders.pop(0)

                # Remove price level if no orders left
                if not buy_orders:
                    del self.buy_orders[buy_price]
                if not sell_orders:
                    del self.sell_orders[sell_price]

        return executed_trades

    def remove_all_orders(self):
        """Remove all remaining orders at the end of the trading day."""
        self.buy_orders.clear()
        self.sell_orders.clear()

    def get_order_book_snapshot(self):
        """Return a snapshot of the current order book."""
        buy_snapshot = {price: sum(order["size"] for order in orders) for price, orders in self.buy_orders.items()}
        sell_snapshot = {price: sum(order["size"] for order in orders) for price, orders in self.sell_orders.items()}
        return buy_snapshot, sell_snapshot

    def plot_order_book(self, market_maker_bid=None, market_maker_bid_volume=None, 
                        market_maker_ask=None, market_maker_ask_volume=None, title="Order Book Snapshot"):
        """Plot the order book showing bid and ask prices with volumes."""
        buy_snapshot, sell_snapshot = self.get_order_book_snapshot()

        # Sort the prices
        buy_prices = sorted(buy_snapshot.keys())
        sell_prices = sorted(sell_snapshot.keys())

        # Volumes corresponding to the sorted prices
        buy_volumes = [buy_snapshot[price] for price in buy_prices]
        sell_volumes = [sell_snapshot[price] for price in sell_prices]

        plt.figure(figsize=(10, 6))

        # Plot Bids (Buy orders)
        plt.bar(buy_prices, buy_volumes, color='green', alpha=0.5, width=0.02, label='Bids', align='center')

        # Plot Asks (Sell orders)
        plt.bar(sell_prices, sell_volumes, color='red', alpha=0.5, width=0.02, label='Asks', align='center')

        # Plot Market Maker's Bid
        if market_maker_bid is not None and market_maker_bid_volume is not None:
            plt.bar(market_maker_bid, market_maker_bid_volume, color='blue', alpha=0.7, width=0.02, label='Market Maker Bid', align='center')

        # Plot Market Maker's Ask
        if market_maker_ask is not None and market_maker_ask_volume is not None:
            plt.bar(market_maker_ask, market_maker_ask_volume, color='orange', alpha=0.7, width=0.02, label='Market Maker Ask', align='center')

        plt.xlabel('Price')
        plt.ylabel('Volume')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

# Analysis Class
class Analysis:
    def __init__(self):
        self.data = []

    def collect_data(self, agent, current_price, is_market_maker=False):
        self.data.append({
            "agent_id": agent.id,
            "cash": agent.cash,
            "inventory": agent.inventory,
            "current_price": current_price,
            "is_market_maker": is_market_maker
        })

    def create_dataframe(self):
        df = pd.DataFrame(self.data)
        return df

    def analyze_profit(self, df):
        initial_cash = df.groupby('agent_id')['cash'].first()
        df['PnL'] = df['cash'] + df['inventory'] * df['current_price'] - initial_cash[df['agent_id']].values
        return df

    def plot_results(self, df):
        market_maker_data = df[df['is_market_maker']]

        # Create a figure and two subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # Plot PnL in the first subplot
        axs[0].plot(market_maker_data.index, market_maker_data['PnL'], label='Market Maker PnL', color='red')
        axs[0].set_title('Market Maker PnL Over Time')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('PnL')
        axs[0].legend()
        axs[0].grid(True)

        # Plot Inventory in the second subplot
        axs[1].plot(market_maker_data.index, market_maker_data['inventory'], label='Market Maker Inventory', color='blue')
        axs[1].set_title('Market Maker Inventory Over Time')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Inventory')
        axs[1].legend()
        axs[1].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

def pause_and_resume():
    while True:
        user_input = input("Press 1 to continue or 2 to break: ")  # Prompt inside the loop
        if user_input == '1':  # Check against the string '1'
            print("Continuing...")
            break  # This will exit the loop and continue the program
        elif user_input == '2':  # Check against the string '2'
            print("Exiting program...")
            exit()  # Exit the entire program
        else:
            print("Invalid input. Please press 1 to continue or 2 to break.")

# Simulation Setup
if __name__ == "__main__":

    # Simulate Stock Prices
    seed=50
    simulator = PriceSimulator(initial_price=100, mu=0.0001, sigma=0.25, dt=1, seed=seed)
    prices = simulator.simulate_brownian_motion_prices()
    price_df = simulator.create_dataframe(prices)
    
    # Agents Setup
    np.random.seed(seed)
    HEDGE_FUNDS = 1
    MARKET_MAKER = 1    
    GAMBLING_AGENTS = 1000 - HEDGE_FUNDS - MARKET_MAKER # Subtract 5 Hedge Funds and 1 Market Maker
    
    aggressiveness_values = np.random.uniform(0, 0.2, GAMBLING_AGENTS)

    gamblers = [GamblerAgent(id=i+1, initial_cash=100_000, initial_inventory=10_000, aggressiveness=aggressiveness_values[i]) for i in range(GAMBLING_AGENTS)]
    hedge_funds = [HedgeFundAgent(id=i+GAMBLING_AGENTS, initial_cash=100_000, initial_inventory=0, aggressiveness=random.choice([0.3, 0.4])) for i in range(HEDGE_FUNDS)]
    market_maker = MarketMaker(id=GAMBLING_AGENTS+HEDGE_FUNDS+1, initial_cash=10_000_000, initial_inventory=0, risk_aversion=0.1)
    
    agents = gamblers + hedge_funds

    # Initialize OrderBook and Analysis
    order_book = OrderBook()
    analysis = Analysis()

    # Run the Simulation
    window_size = 480 * 5  # Last 5 days of trading

    for t, price in enumerate(prices):
        if GAMBLING_AGENTS <= 0:
            print("Gamblers Bankrupt - Simulation Finished")
            break
        if t < window_size:
            continue  # Wait until we have enough data for the gamblers

        # Future price is simply the next price in the simulation
        future_price = prices[t + 100] if t + 1 < len(prices) else price
        recent_prices = prices[t-window_size:t]  # Last 5 days

        print(f"Time: {t}, Current Price:{recent_prices[-1]}, Next Tick Price: {future_price}")

        for agent in agents:
            if agent.is_liquidated:
                continue
            # else:
                # print(f"Gambling Agent {agent.id} has got {agent.cash} left")
            order = None  # Initialize order to None for each agent
            
            if isinstance(agent, GamblerAgent):
                order = agent.place_order(recent_prices, current_time=t)
                # print(f"Gambler Agent ID {agent.id} Order -> Position: {order[0]}, Price: {order[1]}, Volume: {order[2]}")
            elif isinstance(agent, HedgeFundAgent):
                order = agent.place_order(price, future_price, current_time=t)
                # print(f"Hedge Fund Agent ID {agent.id} Order -> Position: {order[0]}, Price: {order[1]}, Volume: {order[2]}")
            # Check if the current agent's order is valid and add to the order book
            if order:
                # print(f"Agent ID {agent.id} Order -> Position: {order[0]}, Price: {order[1]}, Volume: {order[2]}, Aggression: {agent.aggressiveness}")
                order_book.add_order(agent.id, order[0], order[1], size=order[2], timestamp=t)

        # Market Maker places orders based on its strategy
        bid, bid_volume, ask, ask_volume = market_maker.place_order(current_price=price, order_book=order_book, inventory=market_maker.inventory)
        order_book.add_order(market_maker.id, "buy", bid, size=bid_volume, timestamp=t)
        order_book.add_order(market_maker.id, "sell", ask, size=ask_volume, timestamp=t)
        print(f"Market Maker Order: Bid: {bid}, Bid Volume: {bid_volume}, Ask: {ask}, Ask Volume: {ask_volume}")
        order_book.plot_order_book(market_maker_bid=bid,market_maker_bid_volume=bid_volume,market_maker_ask=ask,
                                   market_maker_ask_volume=ask_volume,title=f"Order Book Snapshot at T={t}")
        # pause_and_resume()
        # Execute orders
        executed_orders = order_book.execute_trades(current_price=price, agents=agents,timestamp=t)

        # Adjust inventory and cash based on executed orders
        for order in executed_orders:
            agent = market_maker if order['buyer_id'] == market_maker.id or order['seller_id'] == market_maker.id else agents[order['buyer_id'] - 1]
            if agent.is_liquidated:
                continue
            if order['buyer_id'] == agent.id:
                agent.inventory += order['size']
                agent.cash -= order['price'] * order['size']
            if order['seller_id'] == agent.id:
                agent.inventory -= order['size']
                agent.cash += order['price'] * order['size']
            if agent.cash < 0:
                agent.is_liquidated = True  # Mark agent as liquidated
                # print(f"{agent}  has been liquidated at time {t}")
                GAMBLING_AGENTS -= 1
                print(f"Gamblers Left: {GAMBLING_AGENTS}")
                if GAMBLING_AGENTS <= 0:
                    print("Gamblers Bankrupt - Simulation Finished")
                    break

        # Collect data for analysis
        for agent in agents:
            analysis.collect_data(agent, price)
        analysis.collect_data(market_maker, price, is_market_maker=True)

        # Optionally plot the order book at specific time intervals
        # if t % (480 * 10) == 0:  # Plot every 10 trading days
        #     order_book.plot_order_book(market_maker_bid=bid,
        #                                market_maker_bid_volume=bid_volume,
        #                                market_maker_ask=ask,
        #                                market_maker_ask_volume=ask_volume,
        #                                title=f"Order Book Snapshot at T={t}")

    # Analyze and Plot Results
    # Collect data for analysis
    for agent in agents:
        agent.export_metrics()
    market_maker.export_metrics()
    df = analysis.create_dataframe()
    df = analysis.analyze_profit(df)
    df.to_csv("data/analysis.csv", index=False)
    market_maker.run_all_analyses()
    analysis.plot_results(df)
