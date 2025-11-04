import matplotlib.pyplot as plt
import numpy as np
import os

def plot_sigma_sequence(sigs, stopping_index, log_filename, save_directory="modules/sd_simple_kes/image_generation_data", show_plot=False):
        """
        Plot the sigma sequence and mark the early stopping point.

        Parameters:
        - sigs: The sigma tensor or numpy array (can be truncated if stopping early).
        - stopping_index: The step index where early stopping was triggered.
        - log_filename: The filename of the generation log (used to match the graph name).
        - save_directory: The folder where the plot should be saved.
        - show_plot: Set to True to display the plot interactively.
        """

        # Extract base name to match log filename
        base_filename = os.path.splitext(os.path.basename(log_filename))[0]
        graph_filename = f"{base_filename}_sigma_plot.png"
        graph_path = os.path.join(save_directory, graph_filename)

        # Prepare sigma sequence for plotting
        sigs_np = sigs.cpu().numpy() if hasattr(sigs, 'cpu') else np.array(sigs)
        x = np.arange(len(sigs_np))

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(x, sigs_np, label='Sigma Sequence', marker='o')
        plt.axvline(x=stopping_index, color='red', linestyle='--', label=f'Stopping Point: {stopping_index}')
        plt.xlabel('Step Index')
        plt.ylabel('Sigma Value')
        plt.title('Sigma Sequence with Early Stopping Point')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(graph_path)

        if show_plot:
            plt.show()

        plt.close()
        return graph_path