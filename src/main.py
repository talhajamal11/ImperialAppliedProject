import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class PriceSimulator:
    def __init__(self, initial_price=100, mu=0.0001, sigma=0.2, days=252, minutes_per_day=480, dt=1, seed=4):
        self.initial_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.days = days
        self.minutes_per_day = minutes_per_day
        self.T = self.days * self.minutes_per_day  # Total minutes for simulation
        self.dt = dt
        self.seed = seed
        self.N = int(self.T / self.dt)  # Number of time steps (should equal T)
        self.brownian_motion_prices = np.zeros(self.N)

    def simulate_brownian_motion_prices(self):
        """ Simulate Prices via Brownian Motion SDE via Euler Maruyama Method
        Returns:
            np.array: Minute by Minute Tick Data for 252 days of trading with 8 hours of Trading each day
        """
        np.random.seed(self.seed)
        self.brownian_motion_prices[0] = self.initial_price
        
        for t in range(1, self.N):
            Z_t = np.random.randn()  # Generate a random normal variable
            self.brownian_motion_prices[t] = self.brownian_motion_prices[t-1] + \
                                             self.sigma * np.sqrt(self.dt) * Z_t
        
        return self.brownian_motion_prices
    
    def generate_trading_days(self):
        """Generate trading days for a full year (252 trading days).
        Returns:
            list: A list of trading days for a year
        """
        self.start_date = datetime(2023, 1, 1)
        self.trading_days = []
        while len(self.trading_days) < 252:
            if self.start_date.weekday() < 5:  # Monday to Friday are trading days
                self.trading_days.append(self.start_date)
            self.start_date += timedelta(days=1)

    def generate_time_series(self):
        """Generate time series for each trading day based on the generated trading days.
        Returns:
            list: A list of times for each minute of the trading day
        """
        self.time_series = []
        for day in self.trading_days:
            for minute in range(480):  # 8 hours * 60 minutes
                self.time_series.append(day + timedelta(minutes=minute))

    def create_dataframe(self, prices):
        """Create a DataFrame with date, time, price, and tick_by_tick_return.
        Args:
            prices (np.array): Array of simulated prices.
        Returns:
            pd.DataFrame: DataFrame containing date, time, price, and tick_by_tick_return.
        """
        print("-- Generating Trading Days --")
        self.generate_trading_days()

        print("-- Generating Time Series --")
        self.generate_time_series()

        if len(prices) != len(self.time_series):
            print(f"Length of Time Series: {len(self.time_series)}")
            print(f"Time Series {self.time_series[:5]}")
            print(f"Length of Prices: {len(prices)}")
            raise ValueError(f"Mismatch in lengths: Prices({len(prices)}) vs Time Series({len(self.time_series)})")

        self.price_df = pd.DataFrame({
            'datetime': self.time_series,
            'price': prices
        })

        # Split datetime into date and time
        self.price_df['date'] = self.price_df['datetime'].dt.date
        self.price_df['time'] = self.price_df['datetime'].dt.time
        self.price_df.drop(columns=['datetime'], inplace=True)

        # Calculate tick-by-tick return
        self.price_df['tick_by_tick_return'] = self.price_df['price'].pct_change().fillna(0)
        self.price_df = self.price_df[["date", "time", "price", "tick_by_tick_return"]]
        print(self.price_df.head())
        print(self.price_df.tail())
        return self.price_df
    
    def plot_prices(self, figsize:tuple=(15, 5)):
        plt.figure(figsize=figsize)
        plt.plot(self.price_df["date"],
                 self.price_df["price"],
                 label="Asset Price")
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

class PassiveAgent(Agent):
    def place_order(self, current_price):
        return None

class LimitOrderAgent(Agent):
    def place_order(self, current_price):
        bid_price = current_price - self.aggressiveness
        ask_price = current_price + self.aggressiveness
        self.order_history.append((bid_price, ask_price))
        return (bid_price, ask_price)

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

# OrderBook Class
class OrderBook:
    def __init__(self):
        self.orders = []

    def add_order(self, agent_id, order_type, price, size=1):
        self.orders.append({"agent_id": agent_id, "type": order_type, "price": price, "size": size})

    def execute_orders(self, current_price):
        executed_orders = []
        for order in self.orders:
            if order['type'] == 'buy' and order['price'] >= current_price:
                executed_orders.append(order)
            elif order['type'] == 'sell' and order['price'] <= current_price:
                executed_orders.append(order)

        for order in executed_orders:
            self.orders.remove(order)
        return executed_orders

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
        # Correct the PnL calculation: initial_cash should not be subtracted twice
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
                if agent_id <= 5:  # Only plot a few sample agents
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
                if agent_id <= 5:  # Only plot a few sample agents
                    plt.plot(agent_data.index, agent_data['inventory'], label=f'Agent {agent_id}', linestyle='--', alpha=0.5)

        plt.title('Inventory Over Time')
        plt.xlabel('Time')
        plt.ylabel('Inventory')
        plt.legend()
        plt.grid(True)
        plt.show()


def pause_or_resume():
    print("Program Paused")
    while True:
        user_input = input("Press 1 to Continue or 2 to Stop: ")
        if user_input == '1':
            print("Program Resumed")
            break
        elif user_input == '2':
            print("Program Stopped")
            return True  # Indicate that the program should stop
        else:
            print("Invalid input. Please press 1 to Continue or 2 to Stop.")


if __name__ == "__main__":
    # Step 1: Simulate Stock Prices
    simulator = PriceSimulator(initial_price=100, mu=0.0001, sigma=0.2, dt=1, seed=4)
    prices = simulator.simulate_brownian_motion_prices()
    price_df = simulator.create_dataframe(prices)
    simulator.plot_prices()


# # Step 2: Generate Limit Order Agents with log-normal distributed aggressiveness
# np.random.seed(42)
# lognormal_mean = np.log(0.5)
# lognormal_std_dev = 0.2

# aggressiveness_values = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_std_dev, size=100)
# aggressiveness_values = np.clip(aggressiveness_values, 0, 1)

# agents = [
#     LimitOrderAgent(id=i+1, initial_cash=100000, initial_inventory=0, aggressiveness=aggressiveness_values[i])
#     for i in range(100)
# ]

# # Step 3: Initialize Market Maker
# market_maker = MarketMaker(id=101, initial_cash=1_000_000, initial_inventory=0, risk_aversion=0.1)

# # Step 4: Initialize OrderBook and Analysis
# order_book = OrderBook()
# analysis = Analysis()

# # Step 5: Run the Simulation
# for t, price in enumerate(prices):
#     for agent in agents:
#         bid, ask = agent.place_order(price)
#         order_book.add_order(agent.id, "buy", bid)
#         order_book.add_order(agent.id, "sell", ask)
    
#     # Market Maker places orders based on its strategy
#     bid, ask = market_maker.place_order(price, market_maker.inventory)
#     order_book.add_order(market_maker.id, "buy", bid)
#     order_book.add_order(market_maker.id, "sell", ask)
    
#     # Execute orders
#     executed_orders = order_book.execute_orders(price)
    
#     # Adjust inventory and cash based on executed orders
#     for order in executed_orders:
#         if order['type'] == 'buy':
#             agent = market_maker if order['agent_id'] == market_maker.id else agents[order['agent_id'] - 1]
#             agent.inventory += order['size']
#             agent.cash -= order['price'] * order['size']
#         elif order['type'] == 'sell':
#             agent = market_maker if order['agent_id'] == market_maker.id else agents[order['agent_id'] - 1]
#             agent.inventory -= order['size']
#             agent.cash += order['price'] * order['size']

#     # Collect data for analysis
#     for agent in agents:
#         analysis.collect_data(agent, price)
#     analysis.collect_data(market_maker, price, is_market_maker=True)

# # Step 6: Analyze and Plot Results
# df = analysis.create_dataframe()
# df = analysis.analyze_profit(df)
# analysis.plot_results(df)
