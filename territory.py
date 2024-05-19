import subprocess
import random
import matplotlib.pyplot as plt
import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import os

class TerritoryManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.territory_map = {}
        self.warp_panels = set()
    
    def add_to_territory(self, pos, agent_id):
        self.territory_map[pos] = agent_id
    
    def is_warp_panel(self, pos):
        return pos in self.warp_panels

    def initialize_warp_panels(self, num_panels):
        while len(self.warp_panels) < num_panels:
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            self.warp_panels.add((x, y))
    
    def get_territory_grid(self):
        territory_grid = np.zeros((self.width, self.height))
        for (x, y), agent_id in self.territory_map.items():
            territory_grid[x, y] = agent_id
        return territory_grid

class RandomWalker(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.territory = set()

    def step(self):
        self.random_move()

    def random_move(self):
        pass

class NormalAgent(RandomWalker):
    def random_move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        if not self.model.territory_manager.is_warp_panel(new_position):
            self.territory.add(new_position)
            self.model.territory_manager.add_to_territory(new_position, self.unique_id)

class DoubleStepAgent(RandomWalker):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.move_counter = 0

    def random_move(self):
        self.move_counter += 1
        if self.move_counter % 6 == 0:
            return
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        if random.random() < 0.7:
            possible_steps = self.model.grid.get_neighborhood(new_position, moore=True, include_center=False)
            new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        if not self.model.territory_manager.is_warp_panel(new_position):
            self.territory.add(new_position)
            self.model.territory_manager.add_to_territory(new_position, self.unique_id)

class TerritoryExpanderAgent(RandomWalker):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.move_counter = 0

    def random_move(self):
        self.move_counter += 1
        if self.move_counter % 3 == 0:
            return
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        if not self.model.territory_manager.is_warp_panel(new_position):
            self.territory.add(new_position)
            self.model.territory_manager.add_to_territory(new_position, self.unique_id)
            for neighbor in self.model.grid.get_neighborhood(new_position, moore=False, include_center=False):
                self.territory.add(neighbor)
                self.model.territory_manager.add_to_territory(neighbor, self.unique_id)

class RandomTeleportAgent(RandomWalker):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.move_counter = 0
    
    def random_move(self):
        self.move_counter += 1
        if self.move_counter % 5 == 0:
            new_position = (self.random.randrange(self.model.grid.width), self.random.randrange(self.model.grid.height))
        
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        if not self.model.territory_manager.is_warp_panel(new_position):
            self.territory.add(new_position)
            self.model.territory_manager.add_to_territory(new_position, self.unique_id)

class RandomWalkModel(Model):
    def __init__(self, width, height, num_agents, num_warp_panels):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.territory_manager = TerritoryManager(width, height)
        self.territory_manager.initialize_warp_panels(num_warp_panels)

        self.datacollector = DataCollector(
            agent_reporters={"Territory": lambda a: list(a.territory)}
        )

        for i in range(self.num_agents):
            agent_type = random.choice([NormalAgent, DoubleStepAgent, TerritoryExpanderAgent, RandomTeleportAgent])
            a = agent_type(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            self.territory_manager.add_to_territory((x, y), i)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# Parameters
grid_width = 100
grid_height = 100
total_steps = 10000
num_agents = 20
territory_update_interval = 100
num_warp_panels = 20

# Create the model
model = RandomWalkModel(grid_width, grid_height, num_agents, num_warp_panels)

# Create the directory for saving the maps if it doesn't exist
if not os.path.exists('territory_map'):
    os.makedirs('territory_map')

# Run the model for total_steps steps
for step in range(total_steps):
    model.step()

    # Update territory visualization every territory_update_interval steps
    if step % territory_update_interval == 0:
        territory_grid = model.territory_manager.get_territory_grid()

        plt.figure(figsize=(10, 8))
        plt.imshow(territory_grid, interpolation='nearest', cmap='tab20')
        plt.title(f'Territory Map at Step {step:04d}')
        plt.colorbar()
        plt.savefig(f'territory_map/territory_map_step_{step:04d}.png')
        plt.close()

        print(f"Step {step} completed.")

# Calculate and print top 100 agents with the largest territories
agent_territories = model.datacollector.get_agent_vars_dataframe().reset_index()
agent_territories['Territory Size'] = agent_territories['Territory'].apply(len)
top_100_agents = agent_territories.groupby('AgentID')['Territory Size'].max().nlargest(100)
print("Top 100 Agents with Largest Territories:")
for agent_id, count in top_100_agents.items():
    agent = next(a for a in model.schedule.agents if a.unique_id == agent_id)
    agent_type = type(agent).__name__
    print(f"Agent {agent_id} ({agent_type}): {count} cells")

print("Simulation complete.")

def create_video_from_images():
    subprocess.run([
        'ffmpeg', '-framerate', '5','-pattern_type','glob', '-i', 'territory_map/territory_map_step_*.png',
        'territory_simulation.gif'
    ])

create_video_from_images()