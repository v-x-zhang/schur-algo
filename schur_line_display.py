"""
Focused Schur Process GUI - Final Row Line Tracking
===================================================

This GUI focuses specifically on visualizing the final row (Schur process):
1. Parameter input for X and Y values
2. Generating Schur process samples
3. Interactive visualization of how each part in the final row evolves
4. Grid visualization available in a separate tab

Features:
- Simple parameter input with validation
- Main view: Line tracking of final row parts evolution
- Secondary tab: Full grid visualization
- Interactive part selection and column navigation
- Clear, focused interface on the final Schur process
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from schur_sampler import sample_push_block_grid, sample_rsk_grid

class SchurLineTracker:    
    def __init__(self, root):
        self.root = root
        self.root.title("Schur Process Sampler")
        self.root.geometry("1200x800")
        
        # Data
        self.current_grid = None
        self.current_X = None
        self.current_Y = None
        
        self.setup_gui()
        self.load_defaults()
        
    def setup_gui(self):
        """Setup the main GUI layout."""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel: Controls
        self.create_controls(main_frame)
        
        # Right panel: Visualization
        self.create_visualization(main_frame)
        
    def create_controls(self, parent):
        """Create the control panel."""
        
        controls_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        controls_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
          # Title
        title_label = ttk.Label(controls_frame, text="Schur Process Sampler", 
                               font=("Arial", 12, "bold"), justify=tk.CENTER)
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Description
        desc_label = ttk.Label(controls_frame, 
                              text="Focus: Evolution of parts in the final row\n(the actual Schur process)", 
                              font=("Arial", 9), justify=tk.CENTER, foreground="blue")
        desc_label.grid(row=1, column=0, columnspan=2, pady=(0, 15))        # Parameters
        ttk.Label(controls_frame, text="X values:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W)
        ttk.Label(controls_frame, text="(supports equations, e.g., [0.1*2**i for i in range(3)])", 
                 font=("Arial", 8), foreground="gray").grid(row=3, column=0, columnspan=2, sticky=tk.W)
        self.x_entry = tk.Text(controls_frame, width=30, height=3, wrap=tk.WORD, font=("Consolas", 10))
        self.x_entry.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        self.x_entry.bind('<KeyRelease>', self.validate_inputs)
        
        ttk.Label(controls_frame, text="Y values:", font=("Arial", 10, "bold")).grid(row=5, column=0, sticky=tk.W)
        ttk.Label(controls_frame, text="(supports equations, e.g., [0.2 + 0.1*i for i in range(2)])", 
                 font=("Arial", 8), foreground="gray").grid(row=6, column=0, columnspan=2, sticky=tk.W)
        self.y_entry = tk.Text(controls_frame, width=30, height=3, wrap=tk.WORD, font=("Consolas", 10))
        self.y_entry.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.y_entry.bind('<KeyRelease>', self.validate_inputs)
          # Validation label
        self.validation_label = ttk.Label(controls_frame, text="", font=("Arial", 9))
        self.validation_label.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Generate button
        self.generate_btn = ttk.Button(controls_frame, text="🎲 Generate Sample", 
                                      command=self.generate_sample, state="disabled")
        self.generate_btn.grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))        # Line tracking controls
        tracking_frame = ttk.LabelFrame(controls_frame, text="Final Row Visualization", padding="5")
        tracking_frame.grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(tracking_frame, text="View Mode:").grid(row=0, column=0, sticky=tk.W)
        self.view_mode_var = tk.StringVar(value="all_parts")
        view_mode_combo = ttk.Combobox(tracking_frame, textvariable=self.view_mode_var, 
                                      values=["all_parts", "single_part"], 
                                      width=12, state="readonly")
        view_mode_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        view_mode_combo.bind('<<ComboboxSelected>>', self.update_line_plot)
        ttk.Label(tracking_frame, text="Part (only effects single_part mode):").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.part_var = tk.StringVar(value="1")
        self.part_combo = ttk.Combobox(tracking_frame, textvariable=self.part_var, 
                                      values=["1"], width=12, state="readonly")  # Will be updated dynamically
        self.part_combo.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        self.part_combo.bind('<<ComboboxSelected>>', self.update_line_plot)
        
        # Add visualization style options
        ttk.Label(tracking_frame, text="Style:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.style_var = tk.StringVar(value="lines")
        style_combo = ttk.Combobox(tracking_frame, textvariable=self.style_var, 
                                  values=["lines", "bars", "area"], 
                                  width=12, state="readonly")        
        style_combo.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        style_combo.bind('<<ComboboxSelected>>', self.update_line_plot)
        
        # Add sampling method selector
        ttk.Label(tracking_frame, text="Sampling:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.sampling_var = tk.StringVar(value="push-block")
        sampling_combo = ttk.Combobox(tracking_frame, textvariable=self.sampling_var, 
                                     values=["push-block", "rsk"], 
                                     width=12, state="readonly")
        sampling_combo.grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        sampling_combo.bind('<<ComboboxSelected>>', self.on_sampling_change)

        # Mathematical Presets
        presets_frame = ttk.LabelFrame(controls_frame, text="Mathematical Presets", padding="5")
        presets_frame.grid(row=11, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        presets = [
            ("Geometric Series", "[0.1 * (0.8)**i for i in range(4)]", "[0.2 * (0.9)**i for i in range(3)]"),
            ("Arithmetic Series", "[0.1 + 0.05*i for i in range(4)]", "[0.15 + 0.1*i for i in range(3)]"),
            ("Powers of 2", "[0.1 * 2**(-i) for i in range(3)]", "[0.2 * 2**(-i) for i in range(2)]"),
            ("Simple List", "0.2, 0.3, 0.4", "0.3, 0.4"),
            ("Linear Growth", "[0.1 * i for i in range(1, 5)]", "[0.2 + 0.1*i for i in range(2)]"),
            ("Exponential Decay", "[0.1 * math.exp(-i * 0.5) for i in range(5)]", "[0.1 * math.exp(-i * 0.5) for i in range(5)]"),
        ]
        
        for i, (name, x_vals, y_vals) in enumerate(presets):
            btn = ttk.Button(presets_frame, text=name, width=12,
                            command=lambda x=x_vals, y=y_vals: self.load_preset(x, y))
            btn.grid(row=i//2, column=i%2, padx=2, pady=2, sticky=(tk.W, tk.E))
            
        presets_frame.columnconfigure(0, weight=1)
        presets_frame.columnconfigure(1, weight=1)

        # Info display
        info_frame = ttk.LabelFrame(controls_frame, text="Current Sample Info", padding="5")
        info_frame.grid(row=12, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.info_text = tk.Text(info_frame, width=30, height=8, wrap=tk.WORD, 
                                font=("Consolas", 9))
        info_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        controls_frame.rowconfigure(12, weight=1)
    
    def create_visualization(self, parent):
        """Create the visualization panels with tabs."""
        
        # Right side container
        viz_frame = ttk.Frame(parent)
        viz_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Main tab: Part Tracking
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Part Tracking")
        self.main_tab.columnconfigure(0, weight=1)
        self.main_tab.rowconfigure(0, weight=1)
        
        # Main visualization frame
        main_viz_frame = ttk.LabelFrame(self.main_tab, text="Parts Evolution", padding="5")
        main_viz_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        main_viz_frame.columnconfigure(0, weight=1)
        main_viz_frame.rowconfigure(0, weight=1)
        
        self.line_fig = Figure(figsize=(10, 6), dpi=100)
        self.line_ax = self.line_fig.add_subplot(111)
        self.line_canvas = FigureCanvasTkAgg(self.line_fig, main_viz_frame)
        self.line_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Secondary tab: Grid Visualization
        self.grid_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.grid_tab, text="Grid View")
        self.grid_tab.columnconfigure(0, weight=1)
        self.grid_tab.rowconfigure(0, weight=1)
        
        # Grid visualization frame
        grid_frame = ttk.LabelFrame(self.grid_tab, text="Complete Schur Process Grid", padding="5")
        grid_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.rowconfigure(0, weight=1)
        self.grid_fig = Figure(figsize=(10, 6), dpi=100)
        self.grid_ax = self.grid_fig.add_subplot(111)
        self.grid_canvas = FigureCanvasTkAgg(self.grid_fig, grid_frame)
        self.grid_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add click event handler for grid
        self.grid_canvas.mpl_connect('button_press_event', self.on_grid_click)
        
        # Add partition display label
        self.partition_display = ttk.Label(grid_frame, text="Click on any grid cell to view its partition", 
                                          font=("Consolas", 11), foreground="blue", 
                                          background="lightyellow", relief="solid", padding=5)
        self.partition_display.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        grid_frame.rowconfigure(0, weight=1)
        grid_frame.rowconfigure(1, weight=0)
        
        # Initialize empty plots
        self.init_plots()
        
    def init_plots(self):
        """Initialize empty plots."""
        self.line_ax.set_title("Final Row Parts Evolution (No data)")
        self.line_ax.set_xlabel("Column")
        self.line_ax.set_ylabel("Part Value")
        self.line_ax.text(0.5, 0.5, "Generate a sample to see\nfinal row parts evolution", 
                         ha='center', va='center', transform=self.line_ax.transAxes,
                         fontsize=12, alpha=0.7)
        self.line_canvas.draw()
        
        self.grid_ax.set_title("Complete Schur Process Grid (No data)")
        self.grid_ax.set_xlabel("Column")
        self.grid_ax.set_ylabel("Row")
        self.grid_ax.text(0.5, 0.5, "Generate a sample to see\ncomplete grid visualization", 
                         ha='center', va='center', transform=self.grid_ax.transAxes,
                         fontsize=12, alpha=0.7)
        self.grid_canvas.draw()
    def load_defaults(self):
        """Load default parameter values."""
        self.x_entry.insert("1.0", "[0.1 * 2**i for i in range(3)]")
        self.y_entry.insert("1.0", "[0.2 + 0.1*i for i in range(2)]")
        self.validate_inputs()
        
    def load_preset(self, x_vals, y_vals):
        """Load preset values."""
        self.x_entry.delete("1.0", tk.END)
        self.y_entry.delete("1.0", tk.END)
        self.x_entry.insert("1.0", x_vals)
        self.y_entry.insert("1.0", y_vals)
        self.validate_inputs()
    def parse_parameters(self):
        """Parse X and Y parameters, supporting mathematical expressions."""
        import math
        import numpy as np
        
        try:
            x_text = self.x_entry.get("1.0", tk.END).strip()
            y_text = self.y_entry.get("1.0", tk.END).strip()
            
            if not x_text or not y_text:
                raise ValueError("Enter both X and Y values")
            
            # Safe namespace for evaluation
            safe_dict = {
                "__builtins__": {},
                "math": math,
                "np": np,
                "sum": sum,
                "range": range,
                "len": len,
                "float": float,
                "int": int,
                "abs": abs,
                "min": min,
                "max": max,
                "pow": pow,
            }
            
            # Try to evaluate as Python expression first
            try:
                if x_text.startswith('[') and x_text.endswith(']'):
                    X = eval(x_text, safe_dict)
                else:
                    # Try comma-separated values
                    X = [float(x.strip()) for x in x_text.split(",") if x.strip()]
            except:
                # Fallback to comma-separated
                X = [float(x.strip()) for x in x_text.split(",") if x.strip()]
            
            try:
                if y_text.startswith('[') and y_text.endswith(']'):
                    Y = eval(y_text, safe_dict)
                else:
                    # Try comma-separated values
                    Y = [float(y.strip()) for y in y_text.split(",") if y.strip()]
            except:
                # Fallback to comma-separated
                Y = [float(y.strip()) for y in y_text.split(",") if y.strip()]
            
            # Convert to list if numpy array
            if hasattr(X, 'tolist'):
                X = X.tolist()
            if hasattr(Y, 'tolist'):
                Y = Y.tolist()
                
            if not X or not Y:
                raise ValueError("At least one value required for X and Y")
                
            if any(x <= 0 for x in X) or any(y <= 0 for y in Y):
                raise ValueError("All values must be positive")
                
            return X, Y
            
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")
            
    def validate_inputs(self, event=None):
        """Validate input parameters."""
        try:
            X, Y = self.parse_parameters()
            max_product = max(x * y for x in X for y in Y)
            
            if max_product >= 1:
                self.validation_label.config(text=f"❌ Constraint violated: max(X×Y) = {max_product:.3f} ≥ 1", 
                                           foreground="red")
                self.generate_btn.config(state="disabled")
                return False
            else:
                self.validation_label.config(text=f"✓ Valid: max(X×Y) = {max_product:.3f} < 1 → Grid: {len(Y)+1}×{len(X)+1}", 
                                           foreground="green")
                self.generate_btn.config(state="normal")
                return True
                
        except ValueError as e:
            self.validation_label.config(text=f"❌ {str(e)}", foreground="red")
            self.generate_btn.config(state="disabled")
            return False
    
    def generate_sample(self):
        """Generate a new Schur process sample."""
        try:
            if not self.validate_inputs():
                return
                
            X, Y = self.parse_parameters()
            sampling_method = self.sampling_var.get()
            
            self.current_X = X
            self.current_Y = Y
            
            # Use the selected sampling method
            if sampling_method == "rsk":
                self.current_grid = sample_rsk_grid(X, Y)
            else:  # push-block
                self.current_grid = sample_push_block_grid(X, Y)
              # Update visualizations
            self.update_line_plot()
            self.update_grid_plot()
            self.update_info()
            
            # Update part selector with actual max parts
            self.update_part_selector()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate sample:\n{str(e)}")
    
    def update_part_selector(self):
        """Update the part selector dropdown based on actual data."""
        if self.current_grid is None:
            return
        
        # Find maximum part count across all partitions in final row
        final_row = self.current_grid[-1]
        max_parts = 0
        for partition in final_row:
            max_parts = max(max_parts, len(partition._parts))
        
        # If no parts found, still allow part 1
        if max_parts == 0:
            max_parts = 1
        
        # Update dropdown values
        part_values = [str(i) for i in range(1, max_parts + 1)]
        self.part_combo['values'] = part_values
          # Reset to part 1 if current selection is out of range
        current_part = self.part_var.get()
        if current_part not in part_values:
            self.part_var.set("1")
    
    def on_sampling_change(self, event=None):
        """Handle sampling method change - regenerate sample if one exists."""
        if self.current_grid is not None:
            # Regenerate with new sampling method
            self.generate_sample()
    
    def on_grid_click(self, event):
        """Handle click events on the grid visualization."""
        if self.current_grid is None or event.inaxes != self.grid_ax:
            return
        
        # Get click coordinates
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        rows = len(self.current_grid)
        cols = len(self.current_grid[0])
        
        # Convert coordinates to grid indices
        col = int(x)
        row = int(y)  # No flipping - direct coordinate mapping
        
        # Check bounds
        if 0 <= col < cols and 0 <= row < rows:
            partition = self.current_grid[row][col]
            if partition._parts:
                partition_text = str(partition._parts)
                size = sum(partition._parts)
            else:
                partition_text = "[]"
                size = 0
            
            # Update display label
            display_text = f"Position ({row}, {col}): λ = {partition_text} (weight = {size})"
            self.partition_display.config(text=display_text)
        else:
            self.partition_display.config(text="Click on any grid cell to view its partition")
            
    def update_grid_plot(self):
        """Update the grid visualization."""
        if self.current_grid is None:
            return
            
        self.grid_ax.clear()
        
        rows = len(self.current_grid)
        cols = len(self.current_grid[0])
        
        # Create color-coded grid
        for n in range(rows):
            for m in range(cols):
                partition = self.current_grid[n][m]
                size = sum(partition._parts) if partition._parts else 0
                
                # Color based on partition size
                if size == 0:
                    color = 'lightgray'
                    text_color = 'black'
                else:
                    intensity = min(size / 5.0, 1.0)  # Normalize to 0-1
                    color = plt.cm.Blues(0.3 + 0.7 * intensity)
                    text_color = 'white' if intensity > 0.5 else 'black'
                  # Draw cell (no flipping - display in natural order)
                rect = plt.Rectangle((m, n), 1, 1, facecolor=color, 
                                   edgecolor='black', linewidth=1)
                self.grid_ax.add_patch(rect)
                
                # No text overlay - will be shown on click instead
                                   # Highlight the final row (which is now at the top)
        self.grid_ax.axhline(y=rows-1, color='red', linewidth=4, alpha=0.8, 
                           label='Final Row (Schur Process)')
        self.grid_ax.legend(loc='upper right')
        self.grid_ax.set_xlim(0, cols)
        self.grid_ax.set_ylim(0, rows)
        self.grid_ax.set_xlabel("Partition in Sequence")
        self.grid_ax.set_ylabel("Row (n)")
        self.grid_ax.set_title(f"Schur Process Grid ({rows}×{cols}) - Click cells to view partitions")
        
        # Set ticks
        self.grid_ax.set_xticks(range(cols + 1))
        self.grid_ax.set_yticks(range(rows + 1))
        
        # Reset partition display
        if hasattr(self, 'partition_display'):
            self.partition_display.config(text="Click on any grid cell to view its partition")
        
        self.grid_canvas.draw()
    
    def update_line_plot(self, event=None):
        """Update the final row visualization."""
        if self.current_grid is None:
            return
            
        self.line_ax.clear()
        
        # Get the final row (Schur process)
        final_row = self.current_grid[-1]
        cols = len(final_row)
        column_indices = list(range(cols))
        
        view_mode = self.view_mode_var.get()
        style = self.style_var.get()
        if view_mode == "all_parts":
            # Show all parts across all columns in the final row
            # Use matplotlib colormap for gradient from blue to red
            import matplotlib.cm as cm
            
            # Find maximum part number that appears
            max_part = 0
            for m in range(cols):
                partition = final_row[m]
                max_part = max(max_part, len(partition._parts))
            
            if max_part == 0:
                self.line_ax.text(0.5, 0.5, "No parts in sequence", 
                                 ha='center', va='center', transform=self.line_ax.transAxes,
                                 fontsize=12, alpha=0.7)
                self.line_canvas.draw()
                return
            
            # Create color gradient from blue to red
            colormap = cm.get_cmap('coolwarm')  # Blue to red gradient
            colors = [colormap(i / max(1, max_part - 1)) for i in range(max_part)]
            for part_num in range(1, max_part + 1):  # Show ALL parts, not just up to 6
                values = []
                for m in range(cols):
                    partition = final_row[m]
                    values.append(partition.part(part_num))
                if any(v > 0 for v in values):  # Only plot if part appears somewhere
                    color = colors[part_num-1]  # Use gradient color
                    
                    if style == "lines":
                        self.line_ax.plot(column_indices, values, 'o-', linewidth=3, 
                                         markersize=8, label=f'Part {part_num}', 
                                         color=color, markerfacecolor='white', 
                                         markeredgecolor=color, markeredgewidth=2)                    
                    elif style == "bars":
                        bar_width = 0.8 / max_part  # Use actual max_part, not capped value
                        offset = (part_num - 1 - (max_part - 1) / 2) * bar_width
                        self.line_ax.bar([x + offset for x in column_indices], values, 
                                        bar_width, label=f'Part {part_num}', 
                                        color=color, alpha=0.8)
                    else:  # area
                        self.line_ax.fill_between(column_indices, values, alpha=0.6, 
                                                 label=f'Part {part_num}', color=color)
                        self.line_ax.plot(column_indices, values, 'o-', linewidth=2, 
                                         markersize=6, color=color)                                     
            self.line_ax.legend(loc='upper center', fontsize=10)
            title = f"Schur Process"
            
        else:
            # Show single part across all columns
            try:
                part_num = int(self.part_var.get())
            except (ValueError, TypeError):
                part_num = 1
                
            values = []
            for m in range(cols):
                partition = final_row[m]
                values.append(partition.part(part_num))
            
            color = '#1f77b4'
            
            if style == "lines":
                self.line_ax.plot(column_indices, values, 'o-', linewidth=4, markersize=10, 
                                 color=color, markerfacecolor='lightblue', 
                                 markeredgecolor=color, markeredgewidth=3)
            elif style == "bars":
                self.line_ax.bar(column_indices, values, 0.8, color=color, alpha=0.8)
            else:  # area
                self.line_ax.fill_between(column_indices, values, alpha=0.4, color='lightblue')
                self.line_ax.plot(column_indices, values, 'o-', linewidth=3, markersize=8, 
                                 color=color)
            
            title = f"Part {part_num} in Schur Process"
        self.line_ax.set_xlabel("Partition in Sequence", fontsize=12)
        self.line_ax.set_ylabel("Part Value", fontsize=12)
        self.line_ax.set_title(title, fontsize=14, fontweight='bold')
        self.line_ax.grid(True, alpha=0.3)
        
        # Set integer ticks for both modes
        self.line_ax.set_xticks(range(cols))
        
        # Collect all y-values to determine axis range
        all_y_values = set()
        if view_mode == "all_parts":
            # Get all values from all parts
            max_part = 0
            for m in range(cols):
                partition = final_row[m]
                max_part = max(max_part, len(partition._parts))
            
            for part_num in range(1, max_part + 1):
                for m in range(cols):
                    partition = final_row[m]
                    value = partition.part(part_num)
                    all_y_values.add(value)  # Include ALL values, including 0
            
            # Always include 0 for reference
            all_y_values.add(0)
                
        else:  # single_part mode
            for m in range(cols):
                partition = final_row[m]
                value = partition.part(int(self.part_var.get()))
                all_y_values.add(value)
            
            # Always include 0 for reference  
            all_y_values.add(0)
          # Set integer y-axis ticks
        if all_y_values:
            min_val = min(all_y_values)
            max_val = max(all_y_values)
            
            # Create integer ticks that include all actual values
            if min_val == max_val:
                # If all values are the same, show that value plus neighbors
                ticks = [max(0, min_val - 1), min_val, min_val + 1]
            else:
                # Include all integer values in the range, plus some padding
                ticks = list(range(max(0, min_val - 1), max_val + 2))
            
            self.line_ax.set_yticks(ticks)
            # Ensure y-axis never goes below 0
            self.line_ax.set_ylim(max(0, min_val - 0.5), max_val + 0.5)
        else:
            # Fallback if no data
            self.line_ax.set_yticks([0, 1, 2])
            self.line_ax.set_ylim(-0.5, 2.5)
        
        # Add some padding to x-axis
        self.line_ax.set_xlim(-0.5, cols - 0.5)
        
        self.line_canvas.draw()
        
    def update_info(self):
        """Update the information display."""
        if self.current_grid is None:
            return
            
        self.info_text.delete(1.0, tk.END)
        
        rows = len(self.current_grid)
        cols = len(self.current_grid[0])
        
        info = f"SAMPLE INFORMATION\\n"
        info += f"{'='*25}\\n"
        info += f"Grid Size: {rows} × {cols}\\n"
        info += f"Sampling Method: {self.sampling_var.get()}\\n"
        info += f"X parameters: {self.current_X}\\n"
        info += f"Y parameters: {self.current_Y}\\n\\n"
        
        info += f"LAST ROW (Schur Process):\\n"
        info += f"{'-'*25}\\n"
        last_row = self.current_grid[-1]
        for m in range(cols):
            partition = last_row[m]
            if partition._parts:
                info += f"λ[{rows-1},{m}] = {partition._parts}\\n"
            else:
                info += f"λ[{rows-1},{m}] = []\\n"
                
        info += f"\\nGRID SUMMARY:\\n"
        info += f"{'-'*25}\\n"
        
        # Count partition sizes
        size_counts = {}
        for n in range(rows):
            for m in range(cols):
                size = sum(self.current_grid[n][m]._parts)
                size_counts[size] = size_counts.get(size, 0) + 1
                
        for size in sorted(size_counts.keys()):
            info += f"Size {size}: {size_counts[size]} cells\\n"
            
        self.info_text.insert(1.0, info)

def main():
    """Run the final row tracker GUI."""
    root = tk.Tk()
    app = SchurLineTracker(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
