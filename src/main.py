import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PriceSimulator:
    def __init__(self, initial_price=100, mu=0, sigma=0.2, T=1.0, dt=1/252, seed=42):
        self.initial_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.seed = seed

    def simulate(self):
        np.random.seed(self.seed)
        N = int(self.T / self.dt)
        prices = np.zeros(N)
        prices[0] = self.initial_price
        
        for t in range(1, N):
            prices[t] = prices[t-1] * np.exp((self.mu - 0.5 * self.sigma**2) * self.dt + 
                                             self.sigma * np.sqrt(self.dt) * np.random.randn())
        
        return prices


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
        # Calculate bid and ask based on aggressiveness and risk aversion
        bid_price = current_price - self.aggressiveness
        ask_price = current_price + self.aggressiveness
        self.order_history.append((bid_price, ask_price))
        return (bid_price, ask_price)

class MarketMaker(Agent):
    def calculate_optimal_prices(self, current_price, inventory):
        # Assume some form of market impact or intensity function
        lambda_b = max(0, 1 - self.risk_aversion * inventory)
        lambda_a = max(0, 1 + self.risk_aversion * inventory)
        
        # Optimal bid and ask prices based on the reservation price idea
        bid_price = current_price - (1 / self.risk_aversion) * np.log(1 + self.risk_aversion * lambda_b)
        ask_price = current_price + (1 / self.risk_aversion) * np.log(1 + self.risk_aversion * lambda_a)
        
        self.order_history.append((bid_price, ask_price))
        return (bid_price, ask_price)
    
    def place_order(self, current_price, inventory):
        # Uses its own inventory to adjust the bid/ask spread
        return self.calculate_optimal_prices(current_price, inventory)



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

        # Execute trades and update agent inventories
        for order in executed_orders:
            print(f"Executing {order['type']} order for Agent {order['agent_id']} at {order['price']}")
            self.orders.remove(order)
        return executed_orders

class Analysis:
    def __init__(self):
        self.data = []

    def collect_data(self, agent, current_price):
        self.data.append({
            "agent_id": agent.id,
            "cash": agent.cash,
            "inventory": agent.inventory,
            "current_price": current_price,
        })

    def create_dataframe(self):
        df = pd.DataFrame(self.data)
        return df

    def analyze_profit(self, df):
        # Calculate profit and PnL
        df['PnL'] = df['cash'] + df['inventory'] * df['current_price'] - df['cash'].iloc[0]
        return df

    def plot_results(self, df):
        df.groupby('agent_id')['PnL'].plot(legend=True)
        plt.title('Agent PnL Over Time')
        plt.xlabel('Time')
        plt.ylabel('PnL')
        plt.show()

        df.groupby('agent_id')['inventory'].plot(legend=True)
        plt.title('Agent Inventory Over Time')
        plt.xlabel('Time')
        plt.ylabel('Inventory')
        plt.show()


# Initialize the price simulator
price_simulator = PriceSimulator()
prices = price_simulator.simulate()

# Create agents
agents = [
    PassiveAgent(id=1, initial_cash=100000, initial_inventory=0),
    LimitOrderAgent(id=2, initial_cash=100000, initial_inventory=0, aggressiveness=0.5),
    MarketMaker(id=3, initial_cash=100000, initial_inventory=0, risk_aversion=0.1)
]

order_book = OrderBook()
analysis = Analysis()

# Simulate over time
for t, price in enumerate(prices):
    for agent in agents:
        if isinstance(agent, PassiveAgent):
            continue

        elif isinstance(agent, LimitOrderAgent):
            bid, ask = agent.place_order(price)
            order_book.add_order(agent.id, "buy", bid)
            order_book.add_order(agent.id, "sell", ask)

        elif isinstance(agent, MarketMaker):
            bid, ask = agent.place_order(price, agent.inventory)
            order_book.add_order(agent.id, "buy", bid)
            order_book.add_order(agent.id, "sell", ask)

    # Execute orders in the order book
    executed_orders = order_book.execute_orders(price)

    # Adjust inventory and cash based on executed orders
    for order in executed_orders:
        if order['type'] == 'buy':
            agents[order['agent_id'] - 1].inventory += order['size']
            agents[order['agent_id'] - 1].cash -= order['price'] * order['size']
        elif order['type'] == 'sell':
            agents[order['agent_id'] - 1].inventory -= order['size']
            agents[order['agent_id'] - 1].cash += order['price'] * order['size']

    # Collect data for analysis
    for agent in agents:
        analysis.collect_data(agent, price)

# Create DataFrame and analyze results
df = analysis.create_dataframe()
df = analysis.analyze_profit(df)
analysis.plot_results(df)

