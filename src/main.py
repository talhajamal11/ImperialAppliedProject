import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import collections
import random
import sys

# Price Simulation Class
class PriceSimulator:
    def __init__(self, initial_price=100, mu=0.0001, sigma=0.2, days=252, minutes_per_day=480, dt=1, seed=4):
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
        while len(self.trading_days) < 252:
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

    def place_order(self, recent_prices, current_time):
        if self.is_liquidated:
            return None

        # Check if it's time to close a position
        if self.current_position:
            order_type = "sell" if self.current_position['type'] == "buy" else "buy"
            price = round(recent_prices[-1], 2)
            size = round(self.current_position['size'], 2)
            self.current_position = None  # Reset current position after closing
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

            # Calculate order size and price
            price = round(recent_prices[-1] * (1 + (0.01 * self.aggressiveness) * direction), 2)
            order_size = round((self.cash * self.aggressiveness) / price, 2)

            # Ensure the agent can afford the position
            if self.cash < (abs(order_size) * price):
                return None  # Skip the trade if not enough cash

            # Store the order in the order history and update position
            self.current_position = {"type": "buy" if direction > 0 else "sell", "price": price, "size": abs(order_size)}
            self.order_history.append((price, order_size))

            return ("buy" if direction > 0 else "sell", price, abs(order_size))

        return None

    def update_pnl(self, trade_price, trade_size, trade_type):
        if trade_type == "buy":
            self.inventory += trade_size
            self.cash -= trade_price * trade_size
        elif trade_type == "sell":
            self.inventory -= trade_size
            self.cash += trade_price * trade_size

        if self.cash < 0:
            self.is_liquidated = True  # Liquidate if cash is negative

class HedgeFundAgent(Agent):
    def __init__(self, id, initial_cash, initial_inventory, aggressiveness):
        super().__init__(id, initial_cash, initial_inventory, aggressiveness)
        self.aggressiveness = random.choice([0.5, 0.6, 0.7, 0.8, 0.9])  # Each Hedge Fund will be different level of Aggressive
        self.current_position = None  # Track current open position

    def place_order(self, current_price, future_price, current_time):
        if self.is_liquidated:
            return None

        # Open a new position if no current position
        if not self.current_position:
            # Decide to buy or sell based on perfect foresight
            if future_price > current_price:
                # Buy if the future price is higher
                price = round(current_price, 2)
                order_size = round(self.cash * self.aggressiveness, 2)
                # Ensure the agent can afford the position
                if self.cash < order_size * price:
                    return None  # Skip the trade if not enough cash

                # Store the order in the order history and update position
                self.current_position = {"type": "buy", "price": price, "size": order_size}
                self.order_history.append(("buy", price, order_size))
                return ("buy", price, order_size)
            else:
                # Sell if the future price is lower
                price = round(current_price, 2)
                order_size = round(self.inventory * self.aggressiveness, 2)  # Sell part of inventory

                # Ensure the agent has inventory to sell
                if self.inventory < order_size:
                    return None  # Skip the trade if not enough inventory

                # Store the order in the order history and update position
                self.current_position = {"type": "sell", "price": price, "size": order_size}
                self.order_history.append(("sell", price, order_size))
                return ("sell", price, order_size)

        return None

    def update_pnl(self, trade_price, trade_size, trade_type):
        if trade_type == "buy":
            self.inventory += trade_size
            self.cash -= trade_price * trade_size
        elif trade_type == "sell":
            self.inventory -= trade_size
            self.cash += trade_price * trade_size

        if self.cash < 0:
            self.is_liquidated = True  # Liquidate if cash is negative

# Market Maker Class
class MarketMaker(Agent):
    def __init__(self, id, initial_cash, initial_inventory, aggressiveness=0.1, risk_aversion=0.1):
        super().__init__(id, initial_cash, initial_inventory, aggressiveness)
        self.risk_aversion = risk_aversion

    def calculate_optimal_prices(self, current_price, inventory):
        lambda_b = max(0, 1 - self.risk_aversion * inventory)
        lambda_a = max(0, 1 + self.risk_aversion * inventory)
        
        bid_price = current_price - (1 / self.risk_aversion) * np.log(1 + self.risk_aversion * lambda_b)
        ask_price = current_price + (1 / self.risk_aversion) * np.log(1 + self.risk_aversion * lambda_a)
        
        self.order_history.append((bid_price, ask_price))
        return (bid_price, ask_price)
    
    def calculate_optimal_volume(self, current_price, order_book, inventory):
        # Factor 1: Adjust based on inventory
        inventory_factor = max(0.1, 1 - self.risk_aversion * abs(inventory))  # Reduce size with large inventory

        # Extract all buy and sell volumes
        buy_volumes = [order["size"] for orders_at_price in order_book.buy_orders.values() for order in orders_at_price]
        sell_volumes = [order["size"] for orders_at_price in order_book.sell_orders.values() for order in orders_at_price]

        # Combine buy and sell volumes to get the overall depth
        all_volumes = buy_volumes + sell_volumes
        
        if len(all_volumes) > 0:
            depth_factor = min(1, np.mean(all_volumes) / 10000)  # Normalize by some factor, say 10000
        else:
            depth_factor = 0.1  # If no orders, use a minimal depth factor to avoid division by zero

        # Factor 3: Adjust based on wealth and risk aversion
        wealth_factor = max(0.1, self.cash / (self.cash + inventory * current_price)) * (1 - self.risk_aversion)

        # Final optimal volume calculation
        optimal_volume = inventory_factor * depth_factor * wealth_factor * 1000  # Scale by some factor, e.g., 1000
        
        return max(1, round(optimal_volume, 2))  # Ensure the volume is at least 1


    def place_order(self, current_price, order_book, inventory):
        bid_price, ask_price = self.calculate_optimal_prices(current_price, inventory)
        bid_volume = self.calculate_optimal_volume(bid_price, order_book, inventory)
        ask_volume = self.calculate_optimal_volume(ask_price, order_book, inventory)
        
        # Store the order in the order history
        self.order_history.append(("buy", bid_price, bid_volume))
        self.order_history.append(("sell", ask_price, ask_volume))
        
        return bid_price, bid_volume, ask_price, ask_volume

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

                    # Update agents' PnL
                    agents[buy_order["agent_id"]].update_pnl(sell_price, trade_size, "buy")
                    agents[sell_order["agent_id"]].update_pnl(sell_price, trade_size, "sell")

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

    def plot_order_book(self, title="Order Book Snapshot"):
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
        plt.figure(figsize=(10, 6))

        # Plot PnL
        for agent_id in df['agent_id'].unique():
            agent_data = df[df['agent_id'] == agent_id]
            if agent_data['is_market_maker'].iloc[0]:
                plt.plot(agent_data.index, agent_data['PnL'], label=f'Market Maker {agent_id}', linestyle='-', color='red')
            else:
                if agent_id <= 5:
                    plt.plot(agent_data.index, agent_data['PnL'], label=f'Agent {agent_id}', linestyle='--', alpha=0.5)

        plt.title('PnL Over Time')
        plt.xlabel('Time')
        plt.ylabel('PnL')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))

        # Plot Inventory
        for agent_id in df['agent_id'].unique():
            agent_data = df[df['agent_id'] == agent_id]
            if agent_data['is_market_maker'].iloc[0]:
                plt.plot(agent_data.index, agent_data['inventory'], label=f'Market Maker {agent_id}', linestyle='-', color='blue')
            else:
                if agent_id <= 5:
                    plt.plot(agent_data.index, agent_data['inventory'], label=f'Agent {agent_id}', linestyle='--', alpha=0.5)

        plt.title('Inventory Over Time')
        plt.xlabel('Time')
        plt.ylabel('Inventory')
        plt.legend()
        plt.grid(True)
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
    simulator = PriceSimulator(initial_price=100, mu=0.0001, sigma=0.25, dt=1, seed=50)
    prices = simulator.simulate_brownian_motion_prices()
    price_df = simulator.create_dataframe(prices)
    
    # Agents Setup
    np.random.seed(50)
    HEDGE_FUNDS = 0
    MARKET_MAKER = 1    
    GAMBLING_AGENTS = 1000 - HEDGE_FUNDS - MARKET_MAKER # Subtract 5 Hedge Funds and 1 Market Maker
    
    aggressiveness_values = np.random.uniform(0, 1, GAMBLING_AGENTS)

    gamblers = [GamblerAgent(id=i+1, initial_cash=100_000, initial_inventory=0, aggressiveness=aggressiveness_values[i]) for i in range(GAMBLING_AGENTS)]
    hedge_funds = [HedgeFundAgent(id=i+GAMBLING_AGENTS, initial_cash=100_000_000, initial_inventory=0) for i in range(HEDGE_FUNDS)]
    market_maker = MarketMaker(id=GAMBLING_AGENTS+HEDGE_FUNDS+1, initial_cash=10_000_000, initial_inventory=0, risk_aversion=0.1)
    
    agents = gamblers

    # Initialize OrderBook and Analysis
    order_book = OrderBook()
    analysis = Analysis()

    # Run the Simulation
    window_size = 480 * 5  # Last 5 days of trading

    for t, price in enumerate(prices):
        if t < window_size:
            continue  # Wait until we have enough data for the gamblers

        # Future price is simply the next price in the simulation
        future_price = prices[t + 1] if t + 1 < len(prices) else price
        recent_prices = prices[t-window_size:t]  # Last 5 days

        print(f"Time: {t}, Current Price:{recent_prices[-1]}, Next Tick Price: {future_price}")

        for agent in agents:
            order = None  # Initialize order to None for each agent

            if isinstance(agent, GamblerAgent):
                order = agent.place_order(recent_prices, current_time=t)
            
            elif isinstance(agent, HedgeFundAgent):
                order = agent.place_order(price, future_price, current_time=t)
            
            # Check if the current agent's order is valid and add to the order book
            if order:
                print(f"Agent ID {agent.id} Order -> Position: {order[0]}, Price: {order[1]}, Volume: {order[2]}, Aggression: {agent.aggressiveness}")
                order_book.add_order(agent.id, order[0], order[1], size=order[2], timestamp=t)
       
        order_book.plot_order_book()

        # Market Maker places orders based on its strategy
        bid, bid_volume, ask, ask_volume = market_maker.place_order(current_price=price, order_book=order_book, inventory=market_maker.inventory)
        order_book.add_order(market_maker.id, "buy", bid, size=bid_volume, timestamp=t)
        order_book.add_order(market_maker.id, "sell", ask, size=ask_volume, timestamp=t)
        print(f"Market Maker Order: Bid: {bid}, Bid Volume: {bid_volume}, Ask: {ask}, Ask Volume: {ask_volume}")
        
        pause_and_resume()
        
        # Execute orders
        executed_orders = order_book.execute_trades(current_price=price, agents=agents,timestamp=t)

        # Adjust inventory and cash based on executed orders
        for order in executed_orders:
            
            agent = market_maker if order['buyer_id'] == market_maker.id or order['seller_id'] == market_maker.id else agents[order['buyer_id'] - 1]
            
            if order['buyer_id'] == agent.id:
                agent.inventory += order['size']
                agent.cash -= order['price'] * order['size']
            
            if order['seller_id'] == agent.id:
                agent.inventory -= order['size']
                agent.cash += order['price'] * order['size']

            if agent.cash < 0:
                agent.is_liquidated = True  # Mark agent as liquidated
                print(f"{agent}  has been liquidated at time {t}")

        # Collect data for analysis
        for agent in agents:
            analysis.collect_data(agent, price)
        analysis.collect_data(market_maker, price, is_market_maker=True)

        # Optionally plot the order book at specific time intervals
        if t % (480 * 10) == 0:  # Plot every 10 trading days
            order_book.plot_order_book(title=f"Order Book Snapshot at T={t}")

    # # Analyze and Plot Results
    # df = analysis.create_dataframe()
    # df = analysis.analyze_profit(df)
    # analysis.plot_results(df)
