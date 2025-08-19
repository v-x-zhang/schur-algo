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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from schur_sampler import sample_push_block_grid, sample_rsk_grid, sample_rsk_grid_symmetric

class SchurLineTracker:    
    def __init__(self, root):
        self.root = root
        self.root.title("Schur Process Sampler")
        self.root.geometry("1600x800")  # Increased width for wider controls and visualization
        
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
        
        # Configure grid weights - give more space to visualization
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=0)  # Controls: fixed width
        main_frame.columnconfigure(1, weight=1)  # Visualization: expandable
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel: Controls
        self.create_controls(main_frame)
        
        # Right panel: Visualization
        self.create_visualization(main_frame)
        
    def create_controls(self, parent):
        """Create the control panel with scrollbar."""
        
        # Create outer frame for controls - remove padding for more space
        controls_outer = ttk.LabelFrame(parent, text="Controls", padding="0")
        controls_outer.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.N, tk.S), padx=(0, 5))
        controls_outer.columnconfigure(0, weight=0)  # Don't expand
        controls_outer.rowconfigure(0, weight=1)
        
        # Create canvas and scrollbar for scrolling - increased width for dropdown lists
        canvas = tk.Canvas(controls_outer, width=480, highlightthickness=0)
        scrollbar = ttk.Scrollbar(controls_outer, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Create a window in the canvas for the scrollable frame
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        def configure_scroll_region(event=None):
            # Update scroll region and canvas size
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Make sure the canvas window fills the canvas width
            canvas_width = canvas.winfo_width()
            if canvas_width > 1:  # Avoid setting width to 0
                canvas.itemconfig(canvas_window, width=canvas_width)
            # Add extra space at the bottom for better scrolling
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=(bbox[0], bbox[1], bbox[2], bbox[3] + 50))
        
        def configure_canvas(event):
            # Update the scroll region when canvas size changes
            configure_scroll_region()
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_canvas)
        
        # Pack canvas and scrollbar
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure columns for the scrollbar layout
        controls_outer.columnconfigure(0, weight=1)  # Canvas expands
        controls_outer.columnconfigure(1, weight=0)  # Scrollbar fixed width
        
        # Bind mousewheel scrolling to the controls area
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        # Bind mouse wheel events when mouse enters the controls area
        controls_outer.bind('<Enter>', _bind_to_mousewheel)
        controls_outer.bind('<Leave>', _unbind_from_mousewheel)
        
        # Also bind to the canvas itself
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Bind mousewheel scrolling to the entire controls area
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        # Bind mouse wheel events when mouse enters the controls area
        controls_outer.bind('<Enter>', _bind_to_mousewheel)
        controls_outer.bind('<Leave>', _unbind_from_mousewheel)
        
        # Also bind to the canvas itself
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Now use scrollable_frame instead of controls_frame
        controls_frame = scrollable_frame
        
        # Title
        title_label = ttk.Label(controls_frame, text="Schur Process Sampler", 
                               font=("Arial", 12, "bold"), justify=tk.CENTER)
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        # STEP 1: Sampling Algorithm Selection
        sampling_frame = ttk.LabelFrame(controls_frame, text="Step 1: Sampling Algorithm", padding="2")
        sampling_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Label(sampling_frame, text="Algorithm:").grid(row=0, column=0, sticky=tk.W)
        self.sampling_var = tk.StringVar(value="push-block")
        sampling_combo = ttk.Combobox(sampling_frame, textvariable=self.sampling_var, 
                                     values=["push-block", "rsk", "rsk-symmetric"], 
                                     width=20, state="readonly")
        sampling_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        sampling_combo.bind('<<ComboboxSelected>>', self.on_sampling_change)
        
        # Symmetric parameters (shown only for rsk-symmetric)
        self.symmetric_frame = ttk.Frame(sampling_frame)
        
        ttk.Label(self.symmetric_frame, text="q value:").grid(row=0, column=0, sticky=tk.W, pady=(10, 0))
        self.q_var = tk.StringVar(value="0.5")
        self.q_entry = tk.Entry(self.symmetric_frame, textvariable=self.q_var, width=10)
        self.q_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=(10, 0))
        self.q_entry.bind('<KeyRelease>', self.validate_inputs)
        
        ttk.Label(self.symmetric_frame, text="c value:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.c_var = tk.StringVar(value="1.0")
        self.c_entry = tk.Entry(self.symmetric_frame, textvariable=self.c_var, width=10)
        self.c_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        self.c_entry.bind('<KeyRelease>', self.validate_inputs)
        
        # Initially hide symmetric parameters
        self.symmetric_frame.grid_remove()
        
        # STEP 2: Grid Configuration
        grid_config_frame = ttk.LabelFrame(controls_frame, text="Step 2: Grid Configuration", padding="2")
        grid_config_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Variable type selection
        ttk.Label(grid_config_frame, text="Configuration:").grid(row=0, column=0, sticky=tk.W)
        self.config_type_var = tk.StringVar(value="preset")
        config_type_combo = ttk.Combobox(grid_config_frame, textvariable=self.config_type_var,
                                        values=["preset", "custom"],
                                        width=20, state="readonly")
        config_type_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        config_type_combo.bind('<<ComboboxSelected>>', self.on_config_type_change)
        
        # Preset configuration
        self.preset_frame = ttk.Frame(grid_config_frame)
        self.preset_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(self.preset_frame, text="Grid Size:").grid(row=0, column=0, sticky=tk.W)
        self.grid_size_var = tk.StringVar(value="10x10")
        grid_size_combo = ttk.Combobox(self.preset_frame, textvariable=self.grid_size_var,
                                      values=["5x5", "10x10", "25x25", "50x50", "100x100", "250x250", "500x500"],
                                      width=15, state="readonly")
        grid_size_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        grid_size_combo.bind('<<ComboboxSelected>>', self.on_grid_size_change)
        
        ttk.Label(self.preset_frame, text="XY Product Value (Overwritten if using RSK-Symmetric):").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.preset_value_var = tk.StringVar(value="0.5")
        self.preset_value_entry = tk.Entry(self.preset_frame, textvariable=self.preset_value_var, width=15)
        self.preset_value_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        self.preset_value_entry.bind('<KeyRelease>', self.on_preset_value_change)
        
        # Custom configuration
        self.custom_frame = ttk.Frame(grid_config_frame)
        
        ttk.Label(self.custom_frame, text="X values:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(self.custom_frame, text="(e.g., [0.1*2**i for i in range(3)])", 
                 font=("Arial", 8), foreground="gray").grid(row=1, column=0, columnspan=2, sticky=tk.W)
        self.x_entry = tk.Text(self.custom_frame, width=30, height=2, wrap=tk.WORD, font=("Consolas", 9))
        self.x_entry.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        self.x_entry.bind('<KeyRelease>', self.validate_inputs)
        
        ttk.Label(self.custom_frame, text="Y values:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Label(self.custom_frame, text="(e.g., [0.2 + 0.1*i for i in range(2)])", 
                 font=("Arial", 8), foreground="gray").grid(row=4, column=0, columnspan=2, sticky=tk.W)
        self.y_entry = tk.Text(self.custom_frame, width=30, height=2, wrap=tk.WORD, font=("Consolas", 9))
        self.y_entry.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        self.y_entry.bind('<KeyRelease>', self.validate_inputs)
        
        # Initially hide custom frame
        self.custom_frame.grid_remove()
        
        # Validation label
        self.validation_label = ttk.Label(grid_config_frame, text="", font=("Arial", 9))
        self.validation_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Generate button
        self.generate_btn = ttk.Button(controls_frame, text="🎲 Generate Sample", 
                                      command=self.generate_sample, state="disabled")
        self.generate_btn.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Progress bar for sample generation
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(controls_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
        self.progress_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Progress label
        self.progress_label = ttk.Label(controls_frame, text="", font=("Arial", 9))
        self.progress_label.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # STEP 3: Path and Visualization Options
        options_frame = ttk.LabelFrame(controls_frame, text="Step 3: Path & Visualization Options", padding="2")
        options_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Path configuration
        ttk.Label(options_frame, text="Path (optional):").grid(row=0, column=0, sticky=tk.W)
        self.path_entry = tk.Entry(options_frame, width=30, font=("Consolas", 10))
        self.path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        # self.path_entry.bind('<KeyRelease>', self.on_path_change)
        
        ttk.Label(options_frame, text="(D=Down, R=Right)", 
                 font=("Arial", 8), foreground="gray").grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        # Path validation label
        self.path_validation_label = ttk.Label(options_frame, text="", font=("Arial", 9))
        self.path_validation_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(2, 0))
        
        # Path type selection
        ttk.Label(options_frame, text="Path Type:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        self.path_type_var = tk.StringVar(value="custom")
        path_type_combo = ttk.Combobox(options_frame, textvariable=self.path_type_var, 
                                      values=["custom", "diagonal", "right_then_down", "down_then_right", "alternating"], 
                                      width=20, state="readonly")
        path_type_combo.grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=(10, 0))
        # path_type_combo.bind('<<ComboboxSelected>>', self.on_path_type_change)
        
        # Path length
        ttk.Label(options_frame, text="Path Length:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        self.path_length_var = tk.StringVar(value="10")
        path_length_entry = tk.Entry(options_frame, textvariable=self.path_length_var, width=10)
        path_length_entry.grid(row=4, column=1, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        # path_length_entry.bind('<KeyRelease>', self.on_path_type_change)
        
        # Generate path button
        generate_path_btn = ttk.Button(options_frame, text="Generate Path", 
                                      command=self.generate_path)
        generate_path_btn.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Visualization options
        ttk.Label(options_frame, text="Background:").grid(row=6, column=0, sticky=tk.W, pady=(10, 0))
        self.bg_style_var = tk.StringVar(value="white")
        bg_style_combo = ttk.Combobox(options_frame, textvariable=self.bg_style_var, 
                                     values=["white", "colored", "grayscale"], 
                                     width=20, state="readonly")
        bg_style_combo.grid(row=6, column=1, sticky=tk.W, padx=(5, 0), pady=(10, 0))
        bg_style_combo.bind('<<ComboboxSelected>>', self.on_visualization_change)
        
        # Grid partition display toggle
        ttk.Label(options_frame, text="Show Grid Partitions:").grid(row=7, column=0, sticky=tk.W, pady=(10, 0))
        self.show_partitions_var = tk.StringVar(value="fast")
        partition_combo = ttk.Combobox(options_frame, textvariable=self.show_partitions_var,
                                      values=["fast", "detailed"],
                                      width=20, state="readonly")
        partition_combo.grid(row=7, column=1, sticky=tk.W, padx=(5, 0), pady=(10, 0))
        partition_combo.bind('<<ComboboxSelected>>', self.on_visualization_change)
        
        ttk.Label(options_frame, text="(Fast: grid only, Detailed: partition colors)", 
                 font=("Arial", 8), foreground="gray").grid(row=8, column=1, sticky=tk.W, padx=(5, 0))
        
        # Configure column weights
        options_frame.columnconfigure(1, weight=1)
        
        # STEP 4: Graph Configuration
        graph_config_frame = ttk.LabelFrame(controls_frame, text="Step 4: Graph Configuration", padding="2")
        graph_config_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Graph title customization
        ttk.Label(graph_config_frame, text="Custom Title:").grid(row=0, column=0, sticky=tk.W)
        self.custom_title_var = tk.StringVar(value="")
        custom_title_entry = tk.Entry(graph_config_frame, textvariable=self.custom_title_var, width=30)
        custom_title_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        custom_title_entry.bind('<KeyRelease>', self.on_graph_config_change)
        
        ttk.Label(graph_config_frame, text="(Leave empty for auto-generated)", 
                 font=("Arial", 8), foreground="gray").grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        # Axes labels customization
        ttk.Label(graph_config_frame, text="X-axis Label:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.x_axis_label_var = tk.StringVar(value="Step along Path")
        x_axis_entry = tk.Entry(graph_config_frame, textvariable=self.x_axis_label_var, width=30)
        x_axis_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=(10, 0))
        x_axis_entry.bind('<KeyRelease>', self.on_graph_config_change)
        
        ttk.Label(graph_config_frame, text="Y-axis Label:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.y_axis_label_var = tk.StringVar(value="Part Value")
        y_axis_entry = tk.Entry(graph_config_frame, textvariable=self.y_axis_label_var, width=30)
        y_axis_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=(5, 0))
        y_axis_entry.bind('<KeyRelease>', self.on_graph_config_change)
        
        # Tick density control
        ttk.Label(graph_config_frame, text="Step Tick Density:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.tick_density_var = tk.StringVar(value="auto")
        tick_density_combo = ttk.Combobox(graph_config_frame, textvariable=self.tick_density_var,
                                         values=["auto", "sparse", "normal", "dense", "all"],
                                         width=15, state="readonly")
        tick_density_combo.grid(row=4, column=1, sticky=tk.W, padx=(5, 0), pady=(10, 0))
        tick_density_combo.bind('<<ComboboxSelected>>', self.on_graph_config_change)
        
        # Most important: Number of parts to display
        ttk.Label(graph_config_frame, text="Number of Parts to Show:").grid(row=5, column=0, sticky=tk.W, pady=(10, 0))
        self.max_parts_var = tk.StringVar(value="all")
        max_parts_combo = ttk.Combobox(graph_config_frame, textvariable=self.max_parts_var,
                                      values=["all", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                                      width=15, state="readonly")
        max_parts_combo.grid(row=5, column=1, sticky=tk.W, padx=(5, 0), pady=(10, 0))
        max_parts_combo.bind('<<ComboboxSelected>>', self.on_graph_config_change)
        
        ttk.Label(graph_config_frame, text="(Controls which partition parts are plotted)", 
                 font=("Arial", 8), foreground="gray").grid(row=6, column=1, sticky=tk.W, padx=(5, 0))
        
        # Apply settings button
        apply_graph_btn = ttk.Button(graph_config_frame, text="Apply Graph Settings", 
                                    command=self.apply_graph_settings)
        apply_graph_btn.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure column weights
        graph_config_frame.columnconfigure(1, weight=1)
        grid_config_frame.columnconfigure(1, weight=1)
        
        # Add spacer at the bottom for better scrolling
        spacer_frame = ttk.Frame(controls_frame, height=50)
        spacer_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
    
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
        
        # First tab: Grid Visualization
        self.grid_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.grid_tab, text="Grid View")
        self.grid_tab.columnconfigure(0, weight=1)
        self.grid_tab.rowconfigure(0, weight=1)
        
        # Second tab: Path Visualization
        self.path_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.path_tab, text="Path View")
        self.path_tab.columnconfigure(0, weight=1)
        self.path_tab.rowconfigure(0, weight=1)
        
        # Grid visualization frame
        grid_frame = ttk.LabelFrame(self.grid_tab, text="Complete Schur Process Grid", padding="5")
        grid_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.rowconfigure(0, weight=1)
        self.grid_fig = Figure(figsize=(12, 8), dpi=100)  # Increased size for better visualization
        self.grid_ax = self.grid_fig.add_subplot(111)
        self.grid_canvas = FigureCanvasTkAgg(self.grid_fig, grid_frame)
        self.grid_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Path visualization frame
        path_frame = ttk.LabelFrame(self.path_tab, text="Path Partitions", padding="5")
        path_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        path_frame.columnconfigure(0, weight=1)
        path_frame.rowconfigure(0, weight=1)
        
        # Create path plot with navigation
        self.path_fig = Figure(figsize=(12, 8), dpi=100)  # Increased size for better visualization
        # Enable vector format support for high-quality exports
        self.path_fig.set_facecolor('white')
        self.path_ax = self.path_fig.add_subplot(111)
        self.path_canvas = FigureCanvasTkAgg(self.path_fig, path_frame)
        self.path_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add navigation toolbar for path view (zoom, pan, etc.)
        path_toolbar_frame = ttk.Frame(path_frame)
        path_toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        path_toolbar_frame.columnconfigure(0, weight=1)
        
        self.path_toolbar = NavigationToolbar2Tk(self.path_canvas, path_toolbar_frame)
        self.path_toolbar.update()
        
        # Path info display
        self.path_info = tk.Text(path_frame, width=80, height=6, wrap=tk.WORD, 
                                font=("Consolas", 9))
        path_scroll = ttk.Scrollbar(path_frame, orient=tk.VERTICAL, command=self.path_info.yview)
        self.path_info.configure(yscrollcommand=path_scroll.set)
        
        self.path_info.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        path_scroll.grid(row=2, column=1, sticky=(tk.N, tk.S), pady=(5, 0))
        
        path_frame.rowconfigure(0, weight=1)
        path_frame.rowconfigure(1, weight=0)
        path_frame.rowconfigure(2, weight=0)
        
        # Initialize empty plots
        self.init_plots()
        
    def init_plots(self):
        """Initialize empty plots."""
        self.grid_ax.set_title("Complete Schur Process Grid (No data)")
        self.grid_ax.set_xlabel("Column")
        self.grid_ax.set_ylabel("Row")
        self.grid_ax.text(0.5, 0.5, "Generate a sample to see\ncomplete grid visualization", 
                         ha='center', va='center', transform=self.grid_ax.transAxes,
                         fontsize=12, alpha=0.7)
        self.grid_canvas.draw()
        
        self.path_ax.set_title("Path Partitions (No data)")
        self.path_ax.set_xlabel("Step along Path")
        self.path_ax.set_ylabel("Part Value")
        self.path_ax.text(0.5, 0.5, "Enter a path (e.g., 'DDRRDR') to see\npartitions along the path", 
                         ha='center', va='center', transform=self.path_ax.transAxes,
                         fontsize=12, alpha=0.7)
        self.path_canvas.draw()
    def load_defaults(self):
        """Load default parameter values."""
        # Set up default preset configuration
        self.on_grid_size_change()  # This will set up the default 10x10 grid
        self.path_entry.insert(0, "")  # Set a simple default path
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
            
            # For symmetric sampling, we don't need the constraint check
            if self.sampling_var.get() == "rsk-symmetric":
                self.validation_label.config(text=f"✓ Valid for symmetric sampling → Grid: {len(Y)+1}×{len(X)+1}", 
                                           foreground="green")
                self.generate_btn.config(state="normal")
                return True
            
            # For other sampling methods, check the constraint
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
    
    def validate_path(self, event=None):
        """Validate the path parameter."""
        path_text = self.path_entry.get().strip().upper()
        
        if not path_text:
            self.path_validation_label.config(text="", foreground="gray")
            return True
        
        # Check if path contains only D and R characters
        if not all(c in 'DR' for c in path_text):
            self.path_validation_label.config(text="❌ Path must contain only 'D' (Down) and 'R' (Right)", 
                                            foreground="red")
            return False
        
        # Validate path against current grid if available
        if self.current_grid is not None:
            rows = len(self.current_grid)
            cols = len(self.current_grid[0])
            
            # Start at top-left corner of entire grid
            display_row = rows - 1  # Start at the top (final row)
            display_col = 0         # Start at leftmost column
            
            # Check if path stays within entire grid bounds
            for step in path_text:
                if step == 'D':
                    display_row -= 1  # Moving down means decreasing row
                elif step == 'R':
                    display_col += 1  # Moving right
                
                # Check bounds (entire grid is [0, rows-1] x [0, cols-1])
                if display_row < 0 or display_col >= cols:
                    self.path_validation_label.config(
                        text=f"❌ Path goes outside grid bounds ({rows}×{cols})", 
                        foreground="red")
                    return False
            
            self.path_validation_label.config(
                text=f"✓ Valid path: {len(path_text)} steps in {rows}×{cols} grid", 
                foreground="green")
        else:
            self.path_validation_label.config(
                text=f"✓ Valid path format: {len(path_text)} steps", 
                foreground="green")
        
        return True
    
    def on_path_change(self, event=None):
        """Handle path parameter change."""
        self.validate_path()
        if self.current_grid is not None:
            self.update_path_plot()
            self.update_grid_plot()  # Redraw grid with new path
    
    def generate_sample(self):
        """Generate a new Schur process sample."""
        try:
            if not self.validate_inputs():
                return
            
            # Initialize progress bar
            self.progress_var.set(0)
            self.progress_label.config(text="Preparing sample generation...")
            self.generate_btn.config(state="disabled")
            self.root.update()
            
            X, Y = self.parse_parameters()
            sampling_method = self.sampling_var.get()
            
            self.current_X = X
            self.current_Y = Y
            
            # Update progress
            self.progress_var.set(20)
            self.progress_label.config(text=f"Generating sample using {sampling_method}...")
            self.root.update()
            
            # Use the selected sampling method
            if sampling_method == "rsk":
                self.current_grid = sample_rsk_grid(X, Y)
            elif sampling_method == "rsk-symmetric":
                # For symmetric sampling, get c and q parameters
                try:
                    c_val = float(self.c_var.get())
                    q_val = float(self.q_var.get())
                except ValueError:
                    raise ValueError("Invalid c or q value. Please enter valid numbers.")
                
                # Create dummy X and Y arrays for grid sizing
                # The actual values don't matter for symmetric sampling, just the sizes
                M = len(X)
                N = len(Y)
                dummy_X = [0.1] * M  # Small values that satisfy constraint
                dummy_Y = [0.1] * N
                
                self.current_grid = sample_rsk_grid_symmetric(c_val, q_val, dummy_X, dummy_Y)
            else:  # push-block
                self.current_grid = sample_push_block_grid(X, Y)
            
            # Update progress
            self.progress_var.set(60)
            self.progress_label.config(text="Updating grid visualization...")
            self.root.update()
            
            # Update visualizations
            self.update_grid_plot()
            
            # Update progress
            self.progress_var.set(80)
            self.progress_label.config(text="Updating path visualization...")
            self.root.update()
            
            self.update_path_plot()
            
            # Re-validate path with new grid
            self.validate_path()
            
            # Complete
            self.progress_var.set(100)
            self.progress_label.config(text="Sample generation complete!")
            self.generate_btn.config(state="normal")
            self.root.update()
            
            # Clear progress after a short delay
            self.root.after(2000, self.clear_progress)
            
        except Exception as e:
            self.progress_var.set(0)
            self.progress_label.config(text="")
            self.generate_btn.config(state="normal")
            messagebox.showerror("Error", f"Failed to generate sample:\n{str(e)}")
    
    def clear_progress(self):
        """Clear the progress bar and label."""
        self.progress_var.set(0)
        self.progress_label.config(text="")
    
    def on_path_type_change(self, event=None):
        """Handle path type change and auto-generate path if not custom."""
        if self.path_type_var.get() != "custom":
            self.generate_path()
    
    def generate_path(self):
        """Generate a path based on the selected type and length."""
        path_type = self.path_type_var.get()
        
        try:
            length = int(self.path_length_var.get())
        except ValueError:
            length = 10
            self.path_length_var.set("10")
        
        if length <= 0:
            length = 10
            self.path_length_var.set("10")
        
        path = ""
        
        if path_type == "diagonal":
            # Alternate D and R for a diagonal path
            for i in range(length):
                path += "D" if i % 2 == 0 else "R"
        
        elif path_type == "right_then_down":
            # All rights first, then all downs
            rights = length // 2
            downs = length - rights
            path = "R" * rights + "D" * downs
        
        elif path_type == "down_then_right":
            # All downs first, then all rights
            downs = length // 2
            rights = length - downs
            path = "D" * downs + "R" * rights
        
        elif path_type == "alternating":
            # Strict alternating pattern starting with D
            for i in range(length):
                path += "D" if i % 2 == 0 else "R"
        
        elif path_type == "custom":
            # Don't auto-generate for custom
            return
        
        # Set the generated path
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, path)
        
        # Trigger path change event
        self.on_path_change()
    
    def on_sampling_change(self, event=None):
        """Handle sampling method change - regenerate sample if one exists."""
        sampling_method = self.sampling_var.get()
        
        # Show/hide symmetric parameters based on sampling method
        if sampling_method == "rsk-symmetric":
            self.symmetric_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        else:
            self.symmetric_frame.grid_remove()
        
        # Re-validate inputs since constraints change
        self.validate_inputs()
        
        if self.current_grid is not None:
            # Regenerate with new sampling method
            self.generate_sample()
    
    def on_config_type_change(self, event=None):
        """Handle configuration type change between preset and custom."""
        config_type = self.config_type_var.get()
        
        if config_type == "preset":
            self.preset_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
            self.custom_frame.grid_remove()
            # Set default preset values
            self.on_grid_size_change()
        else:  # custom
            self.custom_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
            self.preset_frame.grid_remove()
        
        self.validate_inputs()
    
    def on_grid_size_change(self, event=None):
        """Handle grid size change for preset configuration."""
        grid_size = self.grid_size_var.get()
        value = self.preset_value_var.get()
        
        try:
            q_val = float(value)  # This is the desired XY product
        except ValueError:
            q_val = 0.5
            self.preset_value_var.set("0.5")
        
        if grid_size == "5x5":
            size = 5
        elif grid_size == "10x10":
            size = 10
        elif grid_size == "25x25":
            size = 25
        elif grid_size == "50x50":
            size = 50
        elif grid_size == "100x100":
            size = 100
        elif grid_size == "250x250":
            size = 250
        elif grid_size == "500x500":
            size = 500
        else:
            size = 10
        
        # To achieve XY = q_val, we set X = Y = sqrt(q_val)
        # This way each X[i] * Y[j] = sqrt(q_val) * sqrt(q_val) = q_val
        import math
        sqrt_q = math.sqrt(q_val)
        
        # Update the custom entries with preset values
        x_preset = f"[{sqrt_q} for i in range({size})]"
        y_preset = f"[{sqrt_q} for i in range({size})]"
        
        # Store these for validation
        self.current_preset_x = x_preset
        self.current_preset_y = y_preset
        
        self.validate_inputs()
    
    def on_preset_value_change(self, event=None):
        """Handle preset value change."""
        self.on_grid_size_change()
    
    def on_graph_config_change(self, event=None):
        """Handle graph configuration changes."""
        # Auto-update path plot if it exists
        if self.current_grid is not None and self.path_entry.get().strip():
            self.update_path_plot()
    
    def apply_graph_settings(self):
        """Apply all graph settings and refresh the path plot."""
        if self.current_grid is not None:
            self.update_path_plot()
    
    def on_visualization_change(self, event=None):
        """Handle visualization option changes."""
        if self.current_grid is not None:
            self.update_grid_plot()
    
    def parse_parameters(self):
        """Parse X and Y parameters, supporting mathematical expressions."""
        import math
        import numpy as np
        
        try:
            # Determine if using preset or custom
            if self.config_type_var.get() == "preset":
                x_text = self.current_preset_x if hasattr(self, 'current_preset_x') else "[0.5 for i in range(10)]"
                y_text = self.current_preset_y if hasattr(self, 'current_preset_y') else "[0.5 for i in range(10)]"
            else:
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
    
    def get_path_partitions(self, path_text):
        """Extract partitions along the specified path."""
        if not path_text or self.current_grid is None:
            return []
        
        path_partitions = []
        rows = len(self.current_grid)
        cols = len(self.current_grid[0])
        
        # Start at top-left corner of entire grid
        grid_row = rows - 1     # Start at final row
        grid_col = 0            # Start at leftmost column
        display_row = rows - 1  # Display row (same as grid row)
        display_col = 0         # Display column (same as grid col)
        
        # Add starting partition
        path_partitions.append((display_row, display_col, self.current_grid[grid_row][grid_col]))
        
        # Follow the path
        for step in path_text:
            if step == 'D':
                grid_row -= 1       # Moving down in grid (toward row 0)
                display_row -= 1    # Moving down in display (decreasing y)
            elif step == 'R':
                grid_col += 1       # Moving right in both
                display_col += 1    # Moving right in both
            
            # Check bounds
            if 0 <= grid_row < rows and 0 <= grid_col < cols:
                path_partitions.append((display_row, display_col, self.current_grid[grid_row][grid_col]))
            else:
                break
        
        return path_partitions
    
    def get_path_coordinates(self, path_text):
        """Get the coordinates along the specified path for display."""
        if not path_text:
            return []
        
        coordinates = []
        rows = len(self.current_grid)
        cols = len(self.current_grid[0])
        
        # Start at top-left corner of entire grid
        display_row = rows - 1  # Start at the top (final row)
        display_col = 0         # Start at leftmost column
        coordinates.append((display_row, display_col))
        
        # Follow the path in display coordinates
        for step in path_text:
            if step == 'D':
                display_row -= 1  # Moving down means decreasing row in display
            elif step == 'R':
                display_col += 1  # Moving right
            coordinates.append((display_row, display_col))
        
        return coordinates
    
    def update_grid_plot(self):
        """Update the grid visualization - supports both fast and detailed modes."""
        if self.current_grid is None:
            return
            
        self.grid_ax.clear()
        
        rows = len(self.current_grid)
        cols = len(self.current_grid[0])
        
        # Check if user wants detailed partition visualization
        show_partitions = getattr(self, 'show_partitions_var', None)
        partition_mode = show_partitions.get() if show_partitions else "fast"
        
        if partition_mode == "detailed":
            # Detailed mode: show partition information with colors (slower)
            # First pass: find the global maximum part value for normalization
            max_part_value = 0
            for n in range(rows):
                for m in range(cols):
                    partition = self.current_grid[n][m]
                    if partition._parts:
                        max_part_value = max(max_part_value, max(partition._parts))
            
            # Create color-coded grid
            bg_style = getattr(self, 'bg_style_var', None)
            background_mode = bg_style.get() if bg_style else "colored"
            
            for n in range(rows):
                for m in range(cols):
                    partition = self.current_grid[n][m]
                    
                    if background_mode == "white":
                        # Pure white background for vector-style rendering
                        color = 'white'
                    elif background_mode == "grayscale":
                        # Grayscale based on largest part
                        if not partition._parts:
                            color = 'white'
                        else:
                            largest_part = max(partition._parts)
                            if max_part_value > 0:
                                intensity = largest_part / max_part_value
                            else:
                                intensity = 0
                            gray_value = 0.9 - 0.7 * intensity  # Light gray to dark gray
                            color = (gray_value, gray_value, gray_value)
                    else:  # colored mode
                        # Color based on the largest part in the partition
                        if not partition._parts:
                            color = 'lightgray'
                        else:
                            largest_part = max(partition._parts)
                            if max_part_value > 0:
                                intensity = largest_part / max_part_value  # Normalize to 0-1
                            else:
                                intensity = 0
                            color = plt.cm.Blues(0.3 + 0.7 * intensity)
                    
                    # Draw cell
                    rect = plt.Rectangle((m, n), 1, 1, facecolor=color, 
                                       edgecolor='black', linewidth=0.5)
                    self.grid_ax.add_patch(rect)
        
        else:
            # Fast mode: simple grid outline - much faster
            # Just draw the grid lines
            for i in range(rows + 1):
                self.grid_ax.axhline(y=i, color='black', linewidth=0.5, alpha=0.3)
            for j in range(cols + 1):
                self.grid_ax.axvline(x=j, color='black', linewidth=0.5, alpha=0.3)
            
            # Optional: Fill background with single color
            bg_style = getattr(self, 'bg_style_var', None)
            background_mode = bg_style.get() if bg_style else "white"
            
            if background_mode == "white":
                self.grid_ax.set_facecolor('white')
            elif background_mode == "grayscale":
                self.grid_ax.set_facecolor('lightgray')
            else:
                self.grid_ax.set_facecolor('lightblue')
        
        # Draw path if specified
        path_text = self.path_entry.get().strip().upper()
        if path_text and self.validate_path():
            coordinates = self.get_path_coordinates(path_text)
            if len(coordinates) > 1:
                # Extract x and y coordinates for the path (display coordinates)
                path_x = [coord[1] + 0.5 for coord in coordinates]  # Column + 0.5 for center
                path_y = [coord[0] + 0.5 for coord in coordinates]  # Row + 0.5 for center
                
                # Thin black vector-style line (minimal, clean) - no markers
                self.grid_ax.plot(path_x, path_y, 'k-', linewidth=2, alpha=1.0, 
                                label=f'Path: {len(path_text)} steps')
        
        self.grid_ax.set_xlim(0, cols)
        self.grid_ax.set_ylim(0, rows)
        self.grid_ax.set_xlabel("Partition in Sequence")
        self.grid_ax.set_ylabel("Row (n)")
        
        # Update title based on mode
        mode_text = "Detailed" if partition_mode == "detailed" else "Fast"
        self.grid_ax.set_title(f"Schur Process Grid ({rows}×{cols}) - {mode_text} Mode")
        
        # Set ticks with intelligent spacing to avoid overcrowding
        if cols <= 10:
            x_step = 1
        elif cols <= 25:
            x_step = 5
        elif cols <= 50:
            x_step = 10
        else:
            x_step = 20
            
        if rows <= 10:
            y_step = 1
        elif rows <= 25:
            y_step = 5
        elif rows <= 50:
            y_step = 10
        else:
            y_step = 20
        
        x_ticks = list(range(0, cols + 1, x_step))
        if x_ticks[-1] != cols:  # Always include the last tick
            x_ticks.append(cols)
            
        y_ticks = list(range(0, rows + 1, y_step))
        if y_ticks[-1] != rows:  # Always include the last tick
            y_ticks.append(rows)
        
        self.grid_ax.set_xticks(x_ticks)
        self.grid_ax.set_yticks(y_ticks)
        
        # Disable grid lines for clean vector appearance
        self.grid_ax.grid(False)
        
        self.grid_canvas.draw()
    
    def update_path_plot(self):
        """Update the path visualization."""
        self.path_ax.clear()
        self.path_info.delete(1.0, tk.END)
        
        path_text = self.path_entry.get().strip().upper()
        
        if not path_text or self.current_grid is None:
            self.path_ax.set_title("Path Partitions (No path specified)")
            self.path_ax.text(0.5, 0.5, "Enter a path (e.g., 'DDRRDR') to see\npartitions along the path", 
                             ha='center', va='center', transform=self.path_ax.transAxes,
                             fontsize=12, alpha=0.7)
            self.path_canvas.draw()
            return
        
        if not self.validate_path():
            self.path_ax.set_title("Path Partitions (Invalid path)")
            self.path_ax.text(0.5, 0.5, "Invalid path specification", 
                             ha='center', va='center', transform=self.path_ax.transAxes,
                             fontsize=12, alpha=0.7)
            self.path_canvas.draw()
            return
        
        # Get partitions along the path
        path_partitions = self.get_path_partitions(path_text)
        
        if not path_partitions:
            return
        
        # Get grid dimensions for info text
        rows = len(self.current_grid)
        cols = len(self.current_grid[0])
        
        # Prepare data for plotting
        step_indices = list(range(len(path_partitions)))
        
        # Find maximum number of parts
        max_parts = 0
        for _, _, partition in path_partitions:
            max_parts = max(max_parts, len(partition._parts))
        
        # Apply user's maximum parts setting
        max_parts_setting = self.max_parts_var.get()
        if max_parts_setting != "all":
            try:
                user_max_parts = int(max_parts_setting)
                max_parts = min(max_parts, user_max_parts)
            except ValueError:
                pass  # Keep original max_parts if conversion fails
        
        if max_parts == 0:
            self.path_ax.set_title("Path Partitions (All empty)")
            self.path_ax.text(0.5, 0.5, "All partitions along path are empty", 
                             ha='center', va='center', transform=self.path_ax.transAxes,
                             fontsize=12, alpha=0.7)
            self.path_canvas.draw()
            return
        
        # Plot each part (only up to max_parts)
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        colormap = plt.cm.get_cmap('coolwarm')
        colors = [colormap(i / max(1, max_parts - 1)) for i in range(max_parts)]
        
        for part_num in range(1, max_parts + 1):
            values = []
            for _, _, partition in path_partitions:
                values.append(partition.part(part_num))
            
            if any(v > 0 for v in values):
                # Thin black vector lines without markers
                self.path_ax.plot(step_indices, values, 'k-', linewidth=1, 
                                 label=f'Part {part_num}', alpha=1.0)
        
        # Use custom axis labels if provided
        x_label = self.x_axis_label_var.get() or "Step along Path"
        y_label = self.y_axis_label_var.get() or "Part Value"
        
        self.path_ax.set_xlabel(x_label)
        self.path_ax.set_ylabel(y_label)
        
        # Use custom title if provided, otherwise generate automatic title
        custom_title = self.custom_title_var.get().strip()
        if custom_title:
            title = custom_title
        else:
            # Create a concise title instead of showing the full path
            num_downs = path_text.count('D')
            num_rights = path_text.count('R')
            path_length = len(path_text)
            
            if path_length <= 20:
                # For short paths, show the actual path
                title = f"Path Partitions: {path_text}"
            else:
                # For long paths, show summary statistics
                title = f"Path Partitions ({path_length} steps: {num_downs}D, {num_rights}R)"
        
        self.path_ax.set_title(title)
        self.path_ax.grid(False)  # Remove grid lines for clean vector appearance
        # self.path_ax.legend()
        
        # Set tick spacing based on user preference
        num_steps = len(step_indices)
        tick_density = self.tick_density_var.get()
        
        if tick_density == "all":
            tick_step = 1
        elif tick_density == "dense":
            tick_step = max(1, num_steps // 20)  # About 20 ticks
        elif tick_density == "normal":
            tick_step = max(1, num_steps // 10)  # About 10 ticks
        elif tick_density == "sparse":
            tick_step = max(1, num_steps // 5)   # About 5 ticks
        else:  # "auto" - intelligent spacing
            if num_steps <= 20:
                tick_step = 1
            elif num_steps <= 50:
                tick_step = 5
            elif num_steps <= 100:
                tick_step = 10
            else:
                tick_step = 20
        
        # Select tick indices with chosen spacing
        tick_indices = list(range(0, num_steps, tick_step))
        if tick_indices[-1] != num_steps - 1 and num_steps > 1:
            tick_indices.append(num_steps - 1)  # Always include the last step
        
        self.path_ax.set_xticks(tick_indices)
        
        # Create corresponding labels
        step_labels = []
        for i in tick_indices:
            if i == 0:
                step_labels.append('Start')
            else:
                step_labels.append(f'{path_text[i-1]}{i}')
        
        self.path_ax.set_xticklabels(step_labels, rotation=45, ha='right')
        
        self.path_canvas.draw()
        
        # Update path info
        info_text = f"PATH ANALYSIS\\n"
        info_text += f"{'='*50}\\n"
        info_text += f"Path: {path_text} ({len(path_text)} steps)\\n"
        info_text += f"Grid size: {rows}×{cols} (entire grid accessible)\\n"
        info_text += f"Partitions along path:\\n\\n"
        
        for i, (display_row, display_col, partition) in enumerate(path_partitions):
            step_name = 'Start' if i == 0 else f'{path_text[i-1]}{i}'
            partition_str = str(list(partition._parts)) if partition._parts else '[]'
            weight = sum(partition._parts) if partition._parts else 0
            
            # Display and grid coordinates are now the same
            grid_row = display_row
            grid_col = display_col
            
            info_text += f"Step {i} ({step_name}): Display({display_row},{display_col}) Grid({grid_row},{grid_col}) = {partition_str} (weight={weight})\\n"
        
        self.path_info.insert(1.0, info_text)

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
