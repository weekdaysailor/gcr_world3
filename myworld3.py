import numpy as np
import pyworld3
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class CO2eParameters:
    """Parameters for CO2e tracking and ecological services"""
    base_emission_factor: float  # CO2e per unit of industrial output
    ecological_benefit_delay: int  # Years until ecological benefits take effect
    benefit_effectiveness: float  # Multiplier for ecological services impact

@dataclass
class GCRParameters:
    """Parameters for Global Carbon Reward mechanism"""
    reward_rate: float  # XCC per ton CO2e reduced
    market_adoption_rate: float  # Rate at which XCC is adopted
    initial_xcc_value: float  # Starting value of XCC in standard currency

class BaseWorld3Model:
    """Wrapper for standard pyworld3 model"""
    
    def __init__(self, start_year=1900, end_year=2100):
        self.start_year = start_year
        self.end_year = end_year
        # Create a World3 instance with the specified time range
        self.world3 = pyworld3.World3(
            year_min=self.start_year,
            year_max=self.end_year,
        )
        
    def run_simulation(self):
        """Run standard World3 simulation"""
        self.world3.simulate()  # Changed from run() to simulate()
        return self.get_results()
    
    def get_results(self):
        """Extract relevant results from simulation"""
        results = {
            'year': self.world3.t,  # Changed from year to t
            'population': self.world3.pop,  # Changed from population to pop
            'industrial_output': self.world3.iopc,  # Changed to industrial output per capita
            'pollution': self.world3.ppolx  # Changed to persistent pollution index
        }
        return pd.DataFrame(results)

    
class CO2eTracker:
    """Tracks CO2e emissions and ecological services benefits"""
    
    def __init__(self, params: CO2eParameters):
        self.params = params
        self.emissions_history = []
        self.ecological_benefits = []
        
    def calculate_emissions(self, industrial_output: float) -> float:
        """Calculate CO2e emissions based on industrial output"""
        base_emissions = industrial_output * self.params.base_emission_factor
        ecological_benefit = self.calculate_ecological_benefit()
        return max(0, base_emissions - ecological_benefit)
    
    def calculate_ecological_benefit(self) -> float:
        """Calculate current ecological services benefit"""
        if len(self.emissions_history) < self.params.ecological_benefit_delay:
            return 0
        
        historical_impact = sum(self.emissions_history[-self.params.ecological_benefit_delay:])
        return historical_impact * self.params.benefit_effectiveness

class XCCEconomics:
    """Manages XCC currency dynamics"""
    
    def __init__(self, params: GCRParameters):
        self.params = params
        self.xcc_supply = 0
        self.xcc_value = params.initial_xcc_value
        
    def calculate_reward(self, emission_reduction: float) -> float:
        """Calculate XCC reward for emission reduction"""
        reward = emission_reduction * self.params.reward_rate
        self.xcc_supply += reward
        self.update_xcc_value()
        return reward
    
    def update_xcc_value(self):
        """Update XCC value based on supply and adoption"""
        # Simplified market dynamics
        self.xcc_value *= (1 + self.params.market_adoption_rate - 
                          (0.1 * np.log(1 + self.xcc_supply)))

class GCRWorld3Model(BaseWorld3Model):
    """Extended World3 model with GCR mechanism"""
    
    def __init__(self, co2e_params: CO2eParameters, gcr_params: GCRParameters, 
                 start_year=1900, end_year=2100):
        super().__init__(start_year, end_year)
        self.co2e_tracker = CO2eTracker(co2e_params)
        self.xcc_economics = XCCEconomics(gcr_params)
        
    def run_simulation(self):
        """Run modified World3 simulation with GCR mechanism"""
        self.world3.set_time_range(self.start_year, self.end_year)
        
        # Modify industrial output based on GCR incentives
        original_industrial_output = self.world3.industrial_output_init
        
        def modified_industrial_output(t):
            base_output = original_industrial_output(t)
            emissions = self.co2e_tracker.calculate_emissions(base_output)
            reward = self.xcc_economics.calculate_reward(
                max(0, self.co2e_tracker.emissions_history[-1] - emissions)
                if self.co2e_tracker.emissions_history else 0
            )
            # Adjust output based on XCC incentives
            return base_output * (1 - (reward * self.xcc_economics.xcc_value * 0.001))
        
        self.world3.industrial_output_init = modified_industrial_output
        self.world3.run_simulation()
        return self.get_results()

class SimulationManager:
    """Manages and compares different simulation scenarios"""
    
    def __init__(self, base_model: BaseWorld3Model, gcr_model: GCRWorld3Model):
        self.base_model = base_model
        self.gcr_model = gcr_model
        
    def run_comparisons(self):
        """Run and compare both scenarios"""
        base_results = self.base_model.run_simulation()
        gcr_results = self.gcr_model.run_simulation()
        return base_results, gcr_results
    
    def generate_plots(self, base_results: pd.DataFrame, gcr_results: pd.DataFrame):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Population comparison
        axes[0, 0].plot(base_results['year'], base_results['population'], 
                       label='Baseline')
        axes[0, 0].plot(gcr_results['year'], gcr_results['population'], 
                       label='With GCR')
        axes[0, 0].set_title('Population')
        axes[0, 0].legend()
        
        # Industrial output comparison
        axes[0, 1].plot(base_results['year'], base_results['industrial_output'], 
                       label='Baseline')
        axes[0, 1].plot(gcr_results['year'], gcr_results['industrial_output'], 
                       label='With GCR')
        axes[0, 1].set_title('Industrial Output')
        axes[0, 1].legend()
        
        # Add more plots as needed
        
        plt.tight_layout()
        return fig

def main():
    """Main execution function"""
    # Initialize parameters
    co2e_params = CO2eParameters(
        base_emission_factor=0.5,
        ecological_benefit_delay=10,
        benefit_effectiveness=0.1
    )
    
    gcr_params = GCRParameters(
        reward_rate=1.0,
        market_adoption_rate=0.05,
        initial_xcc_value=100.0
    )
    
    # Create models
    base_model = BaseWorld3Model()
    gcr_model = GCRWorld3Model(co2e_params, gcr_params)
    
    # Run simulations
    sim_manager = SimulationManager(base_model, gcr_model)
    base_results, gcr_results = sim_manager.run_comparisons()
    
    # Generate plots
    fig = sim_manager.generate_plots(base_results, gcr_results)
    plt.show()

if __name__ == "__main__":
    main()
