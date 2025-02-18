import matplotlib
matplotlib.use('QtAgg')
import numpy as np
import matplotlib.pyplot as plt

def calculate_carbon_reward(pollution_reduction, xcc_price):
    return pollution_reduction * xcc_price

def simulate_scenario(population_growth_rate, pollution_generation_rate, pollution_decay_rate,
                      resource_depletion_effect, enable_gcr, co2e_frac=0.65, removal_efficiency=0.15, technology_factor=1.0):
    # Initial conditions
    population = 8.5  # Billion people
    time_steps = 100  # Total number of time steps
    dt = 1  # Time increment (e.g., 1 year)
    industrial_output = 0.5
    industrial_output_growth_rate = 0.03
    co2e = 0.417      # Atmospheric CO2e concentration (ppm)
    other_pollution = 0.2  # Non-GHG pollution
    resources = 1.0   # Fraction of resources remaining
    resource_usage_rate = 0.02
    xcc_price = 80 if enable_gcr else 0
    xcc_growth_rate = 0.1  # 10% per year
    co2e_rate_factor = 1.0  # Initial CO2e rate factor
    co2e_increase_rate = 0.05  # 5% increase per year due to XCC adoption
    resource_availability_multiplier = 1.0  # Initial resource availability multiplier

    # History tracking with correct variable names
    histories = {
        'population': [],
        'industrial_output': [],
        'co2e': [],
        'other_pollution': [],
        'resources': []
    }

    for t in range(time_steps):
        resource_limit = max(0, resources) / (1 + resource_depletion_effect * (1 - resources))
        
        # Store current state
        histories['population'].append(population)
        histories['industrial_output'].append(industrial_output)
        histories['co2e'].append(co2e)
        histories['other_pollution'].append(other_pollution)
        histories['resources'].append(resources)

        # GCR calculations
        if enable_gcr:
            xcc_price *= (1 + xcc_growth_rate * dt)
            co2e_rate_factor *= (1 + co2e_increase_rate * dt)
            mitigation_rate = min((xcc_price * co2e) / industrial_output, 0.15)
            removal_rate = removal_efficiency * xcc_price / 1000
        else:
            mitigation_rate = 0.005
            removal_rate = 0.001

        # Industrial output dynamics
        industrial_output *= (1 + industrial_output_growth_rate * resource_limit * dt) * (1 - mitigation_rate)
        
        # CO2e dynamics (GHGs) with rate factor
        co2e += (
            pollution_generation_rate * industrial_output * co2e_frac * co2e_rate_factor * dt
            - pollution_decay_rate * co2e * dt
            - removal_rate * dt
        )
        
        # Other pollution dynamics
        other_pollution += (
            pollution_generation_rate * industrial_output * (1 - co2e_frac) * dt
            - pollution_decay_rate * other_pollution * dt
        )

        # Resource depletion
        resources -= resource_usage_rate * industrial_output * dt
        
        # Update resource availability multiplier
        resource_availability_multiplier = (resources / (1 + resource_depletion_effect * (1 - resources))) * technology_factor
        
        # Population dynamics
        carrying_capacity = 10.0 * resource_availability_multiplier
        pollution_impact = max(0, 1 - (co2e + other_pollution))
        population *= (1 + population_growth_rate * pollution_impact * 
                      (1 - population / carrying_capacity) * dt)
        
        # Ensure non-negative values
        population = max(0, population)
        industrial_output = max(0, industrial_output)
        co2e = max(0, co2e)
        other_pollution = max(0, other_pollution)
        resources = max(0, resources)

    return histories

# Monte Carlo simulation setup
num_simulations = 100
scenarios = ['with_gcr', 'without_gcr']
result_sets = {scenario: {
    'population': [], 'industrial_output': [], 
    'co2e': [], 'other_pollution': [], 'resources': []
} for scenario in scenarios}

for i in range(num_simulations):
    params = {
        'population_growth_rate': np.random.uniform(-0.01, 0.03),
        'pollution_generation_rate': np.random.normal(0.02, 0.003),
        'pollution_decay_rate': np.random.uniform(0.003, 0.008),
        'resource_depletion_effect': np.random.uniform(4.0, 6.0)
    }

    # Run simulations
    for scenario in scenarios:
        enable_gcr = (scenario == 'with_gcr')
        results = simulate_scenario(enable_gcr=enable_gcr, **params)
        
        for key in result_sets[scenario]:
            result_sets[scenario][key].append(results[key])

# Convert to numpy arrays
for scenario in scenarios:
    for key in result_sets[scenario]:
        result_sets[scenario][key] = np.array(result_sets[scenario][key])

# Plotting function
def plot_scenario(ax, scenario_data, title):
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    labels = ['Population', 'Industrial Output', 'CO2e', 'Other Pollution', 'Resources']
    
    for idx, key in enumerate(['population', 'industrial_output', 'co2e', 'other_pollution', 'resources']):
        mean = np.mean(scenario_data[key], axis=0)
        ax.plot(mean, color=colors[idx], linewidth=2, label=labels[idx])
        ax.fill_between(range(len(mean)),
                        np.percentile(scenario_data[key], 5, axis=0),
                        np.percentile(scenario_data[key], 95, axis=0),
                        color=colors[idx], alpha=0.1)
    
    ax.set_xlabel('Time (Years)')
    ax.set_ylabel('Normalized Values')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

# Create figure

# Updated plotting code with dual axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 18))
colors = ['blue', 'green', 'red', 'orange']
labels = ['Population', 'Industrial Output', 'CO2e', 'Resources']

# Modified plotting section with robust dual-axis handling
def plot_with_dual_axis(ax, scenario_data, title):
    # Create twin axis for population
    ax_pop = ax.twinx()
    
    # Extract data arrays in correct order
    population_data = scenario_data['population']
    industrial_data = scenario_data['industrial_output']
    co2e_data = scenario_data['co2e']
    other_pollution_data = scenario_data['other_pollution']
    resources_data = scenario_data['resources']

    # Plot population on secondary axis (right)
    pop_mean = np.mean(population_data, axis=0)
    ax_pop.plot(pop_mean, color='blue', linewidth=2, label='Population')
    ax_pop.fill_between(range(len(pop_mean)),
                       np.percentile(population_data, 5, axis=0),
                       np.percentile(population_data, 95, axis=0),
                       color='blue', alpha=0.1)
    ax_pop.set_ylabel('Population (Billions)', color='blue')
    ax_pop.tick_params(axis='y', labelcolor='blue')

    # Plot environmental/economic variables on primary axis (left)
    variables = {
        'industrial_output': ('green', 'Industrial Output'),
        'co2e': ('red', 'CO₂e'),
        'other_pollution': ('purple', 'Other Pollution'),
        'resources': ('orange', 'Resources')
    }
    
    for key, (color, label) in variables.items():
        data = scenario_data[key]
        mean = np.mean(data, axis=0)
        ax.plot(mean, color=color, linewidth=2, label=label)
        ax.fill_between(range(len(mean)),
                       np.percentile(data, 5, axis=0),
                       np.percentile(data, 95, axis=0),
                       color=color, alpha=0.1)

    ax.set_xlabel('Time (Years)')
    ax.set_ylabel('Normalized Values')
    ax.set_title(title)
    ax.grid(True)
    
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_pop.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')

# Create figure with updated data structure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 18))
plot_with_dual_axis(ax1, result_sets['with_gcr'], 'With Global Carbon Reward')
plot_with_dual_axis(ax2, result_sets['without_gcr'], 'Without Global Carbon Reward')
plt.tight_layout()
plt.show()
