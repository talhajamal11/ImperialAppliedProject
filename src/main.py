import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import collections

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
        self.brownian_motion_prices[0] = self.initial_price
        
        for t in range(1, self.N):
            Z_t = np.random.randn()
            self.brownian_motion_prices[t] = self.brownian_motion_prices[t-1] + \
                                             self.sigma * np.sqrt(self.dt) * Z_t
        
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
    def __init__(self, id, initial_cash, initial_inventory, risk_aversion=0.1, aggressiveness=0.1):
        self.id = id
        self.cash = initial_cash
        self.inventory = initial_inventory
        self.risk_aversion = risk_aversion
        self.aggressiveness = aggressiveness
        self.order_history = []
        self.is_liquidated = False

class GamblerAgent(Agent):
    def place_order(self, recent_prices):
        if self.is_liquidated:
            return None
        momentum = np.sum(np.diff(recent_prices))
        direction = 1 if momentum > 0 else -1
        order_size = self.cash * self.aggressiveness * direction
        price = recent_prices[-1] * (1 + 0.01 * direction)
        self.order_history.append((price, order_size))
        return ("buy" if direction > 0 else "sell", price, abs(order_size))

class HedgeFundAgent(Agent):
    def place_order(self, current_price, future_price):
        if self.is_liquidated:
            return None

        # Decide to buy or sell based on perfect foresight
        if future_price > current_price:
            # Buy if the future price is higher
            price = current_price
            order_size = self.cash * self.risk_aversion
            self.order_history.append(("buy", price, order_size))
            return ("buy", price, order_size)
        else:
            # Sell if the future price is lower
            price = current_price
            order_size = self.inventory * self.risk_aversion  # Sell part of inventory
            self.order_history.append(("sell", price, order_size))
            return ("sell", price, order_size)

class PassiveAgent(Agent):
    def place_order(self, current_price):
        if self.is_liquidated:
            return None
        return None

# Market Maker Class
class MarketMaker(Agent):
    def calculate_optimal_prices(self, current_price, inventory):
        lambda_b = max(0, 1 - self.risk_aversion * inventory)
        lambda_a = max(0, 1 + self.risk_aversion * inventory)
        
        bid_price = current_price - (1 / self.risk_aversion) * np.log(1 + self.risk_aversion * lambda_b)
        ask_price = current_price + (1 / self.risk_aversion) * np.log(1 + self.risk_aversion * lambda_a)
        
        self.order_history.append((bid_price, ask_price))
        return (bid_price, ask_price)
    
    def place_order(self, current_price, inventory):
        return self.calculate_optimal_prices(current_price, inventory)

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

    def execute_trades(self, current_price, timestamp):
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

        buy_prices = sorted(buy_snapshot.keys(), reverse=True)
        sell_prices = sorted(sell_snapshot.keys())

        buy_volumes = [buy_snapshot[price] for price in buy_prices]
        sell_volumes = [sell_snapshot[price] for price in sell_prices]

        plt.figure(figsize=(10, 6))

        plt.barh(buy_prices, buy_volumes, color='green', alpha=0.5, label='Bids')
        plt.barh(sell_prices, sell_volumes, color='red', alpha=0.5, label='Asks')

        plt.xlabel('Volume')
        plt.ylabel('Price')
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

# Simulation Setup
if __name__ == "__main__":
    # Simulate Stock Prices
    simulator = PriceSimulator(initial_price=100, mu=0.0001, sigma=0.25, dt=1, seed=50)
    prices = simulator.simulate_brownian_motion_prices()
    price_df = simulator.create_dataframe(prices)
    
    # Agent Setup
    np.random.seed(50)
    aggressiveness_values = np.random.uniform(0, 1, 90)
    gamblers = [GamblerAgent(id=i+1, initial_cash=10_000_000, initial_inventory=0, aggressiveness=aggressiveness_values[i]) for i in range(90)]
    hedge_funds = [HedgeFundAgent(id=i+91, initial_cash=100_000_000, initial_inventory=0, risk_aversion=0.05) for i in range(5)]
    passive_agents = [PassiveAgent(id=i+96, initial_cash=5_000_000, initial_inventory=0) for i in range(5)]

    market_maker = MarketMaker(id=101, initial_cash=1_000_000, initial_inventory=0, risk_aversion=0.1)
    
    agents = gamblers + hedge_funds + passive_agents

    print("Prices DataFrame")
    print(simulator.price_df)

    # Initialize OrderBook and Analysis
    order_book = OrderBook()
    analysis = Analysis()

    # # Run the Simulation
    # window_size = 480 * 5  # Last 5 days of trading
    # for t, price in enumerate(prices):
    #     if t < window_size:
    #         continue  # Wait until we have enough data for the gamblers

    #     # Future price is simply the next price in the simulation
    #     future_price = prices[t + 1] if t + 1 < len(prices) else price
        
    #     recent_prices = prices[t-window_size:t]  # Last 5 days
    #     for agent in agents:
    #         if isinstance(agent, GamblerAgent):
    #             order = agent.place_order(recent_prices)
    #         elif isinstance(agent, HedgeFundAgent):
    #             order = agent.place_order(price, future_price)
    #         else:
    #             order = agent.place_order(price)
    #         if order:
    #             order_book.add_order(agent.id, order[0], order[1], size=order[2], timestamp=t)

    #     # Market Maker places orders based on its strategy
    #     bid, ask = market_maker.place_order(price, market_maker.inventory)
    #     order_book.add_order(market_maker.id, "buy", bid, size=1, timestamp=t)
    #     order_book.add_order(market_maker.id, "sell", ask, size=1, timestamp=t)

    #     # Execute orders
    #     executed_orders = order_book.execute_trades(price, timestamp=t)

    #     # Adjust inventory and cash based on executed orders
    #     for order in executed_orders:
    #         agent = market_maker if order['buyer_id'] == market_maker.id or order['seller_id'] == market_maker.id else agents[order['buyer_id'] - 1]
    #         if order['buyer_id'] == agent.id:
    #             agent.inventory += order['size']
    #             agent.cash -= order['price'] * order['size']
    #         if order['seller_id'] == agent.id:
    #             agent.inventory -= order['size']
    #             agent.cash += order['price'] * order['size']

    #         if agent.cash < 0:
    #             agent.is_liquidated = True  # Mark agent as liquidated

    #     # Collect data for analysis
    #     for agent in agents:
    #         analysis.collect_data(agent, price)
    #     analysis.collect_data(market_maker, price, is_market_maker=True)

    #     # Optionally plot the order book at specific time intervals
    #     if t % (480 * 10) == 0:  # Plot every 10 trading days
    #         order_book.plot_order_book(title=f"Order Book Snapshot at T={t}")

    # # Analyze and Plot Results
    # df = analysis.create_dataframe()
    # df = analysis.analyze_profit(df)
    # analysis.plot_results(df)
