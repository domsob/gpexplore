import flet as ft
import gp
import time
import config
import threading
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
import platform

from math import cos, sin, sqrt
from flet.matplotlib_chart import MatplotlibChart
from operator import add, sub, mul, neg
from gp import protectedDiv, analyticQuotient
from copy import deepcopy

matplotlib.use("svg")
warnings.filterwarnings("ignore")

def main(page: ft.Page):

    page.title = "GPExplore v0.1"

    def handle_window_event(e):
        if e.data == "close":
            config.gp_running = False
            page.window_visible = False
            page.window.destroy()

    page.theme_mode = ft.ThemeMode.LIGHT
    # page.window_center()
    page.window_top = page.window_left = 10
    if platform.system() != "Linux":
        page.window_width = 1400
        page.window_height = 950
        page.window_max_width = 1400      
        page.window_max_height = 950       
        page.window_resizable = False
        page.window_maximizable = False
        license_height = 665
    else:
        page.window_width = 1400
        page.window_height = 995
        page.window_max_width = 1400      
        page.window_max_height = 995       
        page.window_resizable = True
        page.window_maximizable = False
        license_height = 654
    page.window.prevent_close = True
    page.window.on_event = handle_window_event

    def target_function_limits_lookup(target_function):
        if target_function == "x**4 + x**3 + x**2 + x":
            return np.linspace(0, 4, 5), -0.6, 4.2, "upper left"
        elif target_function == "x**5 - 2*x**3 + x":
            return np.linspace(-0.3, 0.3, 7), -0.35, 0.35, "lower right"
        elif target_function == "sin(x**2) * cos(x) - 1":
            return np.linspace(-1.0, -0.6, 5), -1.02, -0.5, "upper center"
        elif target_function == "sin(x) + sin(x + x**2)":
            return np.linspace(-1.0, 1.7, 5), -1.02, 1.95, "upper left"
        elif target_function == "sqrt(abs(x))":
            return np.linspace(0, 1.0, 6), -0.01, 1.1, "upper center"
        elif target_function == "x**2":
            return np.linspace(0, 1.0, 6), -0.1, 1.1, "upper center"
        elif target_function == "x**3":
            return np.linspace(-1.0, 1.0, 8), -1.1, 1.1, "lower right"

    def update_main_plots(temp_func=None):
        try:
            config.main_plots = None

            x1 = [] 
            avg_training_y1 = []
            best_training_y1 = []
            avg_validation_y1 = []
            best_validation_y1 = []
            if temp_func != None:
                logbook_temp = deepcopy(config.gp_logbook)

                avg_training_y1 = logbook_temp.chapters["training"].select("avg")
                best_training_y1 = config.best_invididuals_per_generation_fitness
                avg_validation_y1 = logbook_temp.chapters["validation"].select("avg")
                best_validation_y1 = config.best_individuals_per_generation_validation
                x1 = list(range(0, min(len(avg_training_y1), len(best_training_y1))))  

            x2 = config.training_set_x1
            y2 = config.training_set_y1 

            predicted_x2 = np.arange(-1.2, 1.2 + 0.01, 0.01)
            predicted_y2 = None
            if temp_func != None:
                predicted_y2 = [eval(temp_func) for x in predicted_x2]

            plt.figure(figsize=(12, 5), dpi=300, facecolor="none")  

            plt.subplot(1, 2, 1) 
            if len(x1) > 0:
                plt.plot(x1, avg_validation_y1, label="Avg. Validation Error (" + "{:.4f}".format(round(avg_validation_y1[-1], 4)) + ")") 
                plt.plot(x1, avg_training_y1, label="Avg. Training Error (" + "{:.4f}".format(round(avg_training_y1[-1], 4)) + ")", linestyle='dashed') 
                plt.plot(x1, best_validation_y1, label="Best Validation Error (" + "{:.4f}".format(round(best_validation_y1[-1], 4)) + ")") 
                plt.plot(x1, best_training_y1, label="Best Training Error (" + "{:.4f}".format(round(best_training_y1[-1], 4)) + ")", linestyle='dashed') 
                plt.legend(loc='upper right')
            plt.xlabel('Generation')
            plt.ylabel('Mean Squared Error')
            plt.xlim(0, config.generations)
            defined_ticks = [0, math.floor(config.generations / 2), config.generations - 1]   
            plt.xticks(defined_ticks, [str(x+1) for x in defined_ticks])
            if temp_func != None:
                if config.plot_adjust:
                    maximum_y = min(avg_validation_y1[0], 15.0)
                    plt.ylim(-0.04, maximum_y + 0.2)
            else:
                plt.yticks([]) 

            plt.subplot(1, 2, 2) 
            plt.scatter(x2, y2, label='Training Points', color='#FF0101', marker='+')
            if predicted_y2 != None:
                plt.plot(predicted_x2, predicted_y2, label='Predicted Function', color='#0061A4')
            plt.xlabel('X Value')
            plt.ylabel('Y Value')
            if config.plot_adjust:
                y_ticks, min_y_lim, max_y_lim, legend_position = target_function_limits_lookup(config.target_function)
                plt.xticks(np.linspace(-1, 1, 5))
                plt.xlim(-1.1, 1.1)
                plt.yticks(y_ticks)
                plt.ylim(min_y_lim, max_y_lim)
                plt.legend(loc=legend_position)
            else:
                plt.legend()

            plt.tight_layout() 
            # plt.savefig("current_plot.pdf", format="pdf")
            config.main_plots = plt
            return True
        except Exception as e:
            time.sleep(0.2)
            return False 

    def load_gp_data():
        while True:
            if config.gp_running == False:
                break
            if len(config.best_individuals_per_generation) > 0:
                best_solution_textfield.value = config.best_individuals_per_generation[-1]
                successful = update_main_plots(str(config.best_individuals_per_generation[-1]))
                if config.gp_running:
                    current_best_individual_label.value = f"Best Individual (Generation {len(config.best_individuals_per_generation)}):"
                    run_progress_bar.value = ((len(config.best_individuals_per_generation)) / (config.generations))
                if successful:
                    page.update()
            time.sleep(0.1)

    def run_gp(e):
        config.gp_running = True
        config.best_individuals_per_generation = []
        config.best_invididuals_per_generation_fitness = []
        config.best_individuals_per_generation_validation = []
        play_btn.disabled = True
        stop_btn.disabled = False
        target_function_dropdown.disabled = True
        training_set_size_sld.disabled = True
        validation_set_size_sld.disabled = True
        noise_sld.disabled = True
        generations_sld.disabled = True
        population_size_sld.disabled = True
        crossover_probability_sld.disabled = True
        mutation_probability_sld.disabled = True
        disable_function_switches(True)
        selection_method_dropdown.disabled = True
        tourn_size_sld.disabled = True
        initialization_method_dropdown.disabled = True
        min_initial_tree_sld.disabled = True
        max_initial_tree_sld.disabled = True
        max_tree_depth_sld.disabled = True
        plot_adjust_swt.disabled = True
        inter_gen_pause_sld.disabled = True
        page.update()
        # --------------------------------------
        thread0 = threading.Thread(target=gp.run)
        thread0.start()
        thread1 = threading.Thread(target=load_gp_data)
        thread1.start()

    def stop_gp(e):
        config.gp_running = False 
        config.best_individuals_per_generation = []
        config.best_invididuals_per_generation_fitness = []
        config.best_individuals_per_generation_validation = []
        config.gp_logbook = []
        best_solution_textfield.value = ""
        current_best_individual_label.value = f"Best Individual (Generation -):"
        play_btn.disabled = False
        run_progress_bar.value = 0
        update_main_plots()
        stop_btn.disabled = True
        target_function_dropdown.disabled = False
        training_set_size_sld.disabled = False
        validation_set_size_sld.disabled = False
        noise_sld.disabled = False
        generations_sld.disabled = False
        population_size_sld.disabled = False
        crossover_probability_sld.disabled = False
        mutation_probability_sld.disabled = False
        disable_function_switches(False)
        selection_method_dropdown.disabled = False
        tourn_size_sld.disabled = False
        initialization_method_dropdown.disabled = False
        min_initial_tree_sld.disabled = False
        max_initial_tree_sld.disabled = False
        max_tree_depth_sld.disabled = False
        plot_adjust_swt.disabled = False
        inter_gen_pause_sld.disabled = False
        run_progress_bar.value = 0
        page.update()
    
    def training_set_size_slider(e):
        slider_change(False)
        config.training_set_x1 = list(np.random.uniform(-1, 1, config.training_set_size))
        config.training_set_x1.sort()
        config.training_set_y1 = [eval(config.target_function) + np.random.uniform(-config.noise_intensity, config.noise_intensity) for x in config.training_set_x1]
        update_main_plots()
        page.update()

    def validation_set_size_slider(e):
        slider_change(False)
        config.validation_set_x1 = list(np.linspace(-1.1, 1.1, config.validation_set_size))
        page.update()

    def noise_slider(e):
        training_set_size_slider(None)

    def generation_slider(e):
        slider_change(False)
        update_main_plots()
        page.update()

    def slider_change(do_update = True):
        # Main Page:
        training_set_size_label.value = f"Training Set Size ({int(training_set_size_sld.value)}):"
        config.training_set_size = int(training_set_size_sld.value)
        validation_set_size_label.value = f"Validation Set Size ({int(validation_set_size_sld.value)}):"
        config.validation_set_size = int(validation_set_size_sld.value)
        noise_label.value = f"Training Noise Intensity ({round(noise_sld.value, 2)}):"
        config.noise_intensity = round(noise_sld.value, 2)
        population_size_label.value = f"Population Size ({int(population_size_sld.value)}):"
        config.population_size = int(population_size_sld.value)
        generations_label.value = f"Generations ({int(generations_sld.value)}):"
        config.generations = int(generations_sld.value)
        population_size_label.value = f"Population Size ({int(population_size_sld.value)}):"
        config.population_size = int(population_size_sld.value)
        crossover_probability_label.value = f"Crossover Probability ({round(crossover_probability_sld.value, 2)}):"
        config.crossover_probability = round(crossover_probability_sld.value, 2)
        mutation_probability_label.value = f"Mutation Probability ({round(mutation_probability_sld.value, 2)}):"
        config.mutation_probability = round(crossover_probability_sld.value, 2)

        # Config Page(s):
        tourn_size_label.value = f"{int(tourn_size_sld.value)}"
        config.tourn_size = int(tourn_size_sld.value)
        min_initial_tree_label.value = f"{int(min_initial_tree_sld.value)}"
        config.min_initial_tree_depth = int(min_initial_tree_sld.value)
        max_initial_tree_label.value = f"{int(max_initial_tree_sld.value)}"
        config.max_initial_tree_depth = int(max_initial_tree_sld.value)

        if config.min_initial_tree_depth >= config.max_initial_tree_depth:
            config.min_initial_tree_depth = config.max_initial_tree_depth - 1
            min_initial_tree_sld.value = config.min_initial_tree_depth
            min_initial_tree_label.value = f"{int(config.min_initial_tree_depth)}"

        max_tree_depth_label.value = f"{int(max_tree_depth_sld.value)}"
        config.max_tree_depth = int(max_tree_depth_sld.value)
        inter_gen_pause_label.value = f"{round(inter_gen_pause_sld.value, 1)}"
        config.inter_gen_pause = round(inter_gen_pause_sld.value, 1)

        if do_update:
            page.update()

    def changed_target_function(e):
        config.target_function = target_function_dropdown.value
        config.training_set_x1 = list(np.random.uniform(-1, 1, config.training_set_size))
        config.training_set_x1.sort()
        config.training_set_y1 = [eval(config.target_function) + np.random.uniform(-config.noise_intensity, config.noise_intensity) for x in config.training_set_x1]
        config.validation_set_x1 = list(np.linspace(-1.1, 1.1, config.validation_set_size))
        update_main_plots()
        page.update()

    def changed_selection_method(e):
        config.selection_method = selection_method_dropdown.value

    def changed_initialization_method(e):
        config.initialization = initialization_method_dropdown.value

    def get_switches():
        function_switches = [add_func_swt, sub_func_swt, mul_func_swt, protectedDiv_func_swt, analyticQuotient_func_swt, neg_func_swt, 
                             sin_func_swt, cos_func_swt, int101_erc_swt]
        switch_values = [config.add_func, config.sub_func, config.mul_func, config.protectedDiv_func, config.analyticQuotient_func, 
                         config.neg_func, config.sin_func, config.cos_func, config.int101_erc]
        return function_switches, switch_values

    def changed_functions_switch(e):
        function_switches, switch_values = get_switches()
        function_switch_counter = sum([temp.value for temp in function_switches[:-1]])    
        if function_switch_counter == 0:
            open_dlg_modal('Warning!', 'At least one function must always be selected.')
            for switch, value in zip(function_switches, switch_values):
                switch.value = value
            page.update()
        else:
            config.add_func              = add_func_swt.value
            config.sub_func              = sub_func_swt.value
            config.mul_func              = mul_func_swt.value
            config.protectedDiv_func     = protectedDiv_func_swt.value
            config.analyticQuotient_func = analyticQuotient_func_swt.value
            config.neg_func              = neg_func_swt.value
            config.sin_func              = sin_func_swt.value
            config.cos_func              = cos_func_swt.value
            config.int101_erc            = int101_erc_swt.value

    def change_plot_adjust_switch(e):
        config.plot_adjust = plot_adjust_swt.value
        update_main_plots()
        page.update()

    def disable_function_switches(disable=True):
        function_switches, switch_values = get_switches()
        for switch in function_switches:
            if disable:
                switch.disabled = True
            else:
                switch.disabled = False

    def close_dlg(e):
        dlg_modal.open = False
        page.update()
    
    def open_dlg_modal(title, message):
        dlg_title.value = title
        dlg_message.value = message
        page.dialog = dlg_modal
        dlg_modal.open = True
        page.update()

    update_main_plots()

    # Inputs and labels for the main tab
    training_set_size_label = ft.Text(f"Training Set Size ({config.training_set_size}):")
    noise_label = ft.Text(f"Training Noise Intensity ({config.noise_intensity}):")
    validation_set_size_label = ft.Text(f"Validation Set Size ({config.validation_set_size}):")
    generations_label = ft.Text(f"Generations ({config.generations}):")
    population_size_label = ft.Text(f"Population Size ({config.population_size}):")
    crossover_probability_label = ft.Text(f"Crossover Probability ({config.crossover_probability}):")
    mutation_probability_label = ft.Text(f"Mutation Probability ({config.mutation_probability}):")
    current_best_individual_label = ft.Text(f"Best Individual (Generation -):")
    target_function_dropdown = ft.Dropdown(
        value=config.target_function,
        on_change=changed_target_function,
        options=[
            ft.dropdown.Option("x**4 + x**3 + x**2 + x"),
            ft.dropdown.Option("x**5 - 2*x**3 + x"),
            ft.dropdown.Option("sin(x**2) * cos(x) - 1"),
            ft.dropdown.Option("sin(x) + sin(x + x**2)"),
            ft.dropdown.Option("sqrt(abs(x))"),
            ft.dropdown.Option("x**2"),
            ft.dropdown.Option("x**3"),
        ],
    )
    training_set_size_sld = ft.Slider(min=1, max=500, value=config.training_set_size, label="{value}", on_change=slider_change, on_change_end=training_set_size_slider)
    noise_sld = ft.Slider(min=0.0, max=1.0, value=config.noise_intensity, label="{value}", on_change=slider_change, on_change_end=noise_slider)
    validation_set_size_sld = ft.Slider(min=1, max=500, value=config.validation_set_size, label="{value}", on_change=slider_change, on_change_end=validation_set_size_slider)
    generations_sld = ft.Slider(min=1, max=500, value=config.generations, label="{value}", on_change=slider_change, on_change_end=generation_slider)
    population_size_sld = ft.Slider(min=1, max=500, value=config.population_size, label="{value}", on_change=slider_change, on_change_end=slider_change)
    crossover_probability_sld = ft.Slider(min=0.0, max=1.0, value=config.crossover_probability, label="{value}", on_change=slider_change, on_change_end=slider_change)
    mutation_probability_sld = ft.Slider(min=0.0, max=1.0, value=config.mutation_probability, label="{value}", on_change=slider_change, on_change_end=slider_change)
    best_solution_textfield = ft.TextField(
        multiline = True,
        read_only = True,
        min_lines = 12,
        max_lines = 12,
        value = "",
        text_align=ft.TextAlign.LEFT,
        hint_text="Click play to start a GP run!",
    )
    run_progress_bar = ft.ProgressBar(value=0)

    play_btn = ft.ElevatedButton(
        "Start Run", 
        on_click=run_gp, 
        icon="play_arrow",
        style=ft.ButtonStyle(
            padding=ft.Padding(top=5, right=10, bottom=5, left=0),
            shape=ft.RoundedRectangleBorder(radius=5),
            text_style=ft.TextStyle(size=16),
            icon_size=42, 
        ),
        width=150,
        height=50, 
    )
    stop_btn = ft.ElevatedButton(
        "Stop Run", 
        on_click=stop_gp, 
        icon="stop",
        style=ft.ButtonStyle(
            padding=ft.Padding(top=5, right=10, bottom=5, left=0),
            shape=ft.RoundedRectangleBorder(radius=5),
            text_style=ft.TextStyle(size=16),
            icon_size=42, 
        ),
        width=150,
        height=50, 
        disabled=True,
    )

    # Inputs for the function and terminal tab
    add_func_swt = ft.Switch(value=config.add_func, on_change=changed_functions_switch)
    sub_func_swt = ft.Switch(value=config.sub_func, on_change=changed_functions_switch)
    mul_func_swt = ft.Switch(value=config.mul_func, on_change=changed_functions_switch)
    protectedDiv_func_swt = ft.Switch(value=config.protectedDiv_func, on_change=changed_functions_switch)
    analyticQuotient_func_swt = ft.Switch(value=config.analyticQuotient_func, on_change=changed_functions_switch)
    neg_func_swt = ft.Switch(value=config.neg_func, on_change=changed_functions_switch)
    sin_func_swt = ft.Switch(value=config.sin_func, on_change=changed_functions_switch)
    cos_func_swt = ft.Switch(value=config.cos_func, on_change=changed_functions_switch)
    int101_erc_swt = ft.Switch(value=config.int101_erc, on_change=changed_functions_switch)

    # Inputs for the additional settings tab
    toimplement_func_swt = ft.Switch(value="42")
    tourn_size_sld = ft.Slider(min=2, max=9, value=config.tourn_size, divisions=7, width=300, on_change=slider_change, on_change_end=slider_change)
    tourn_size_label = ft.Text(f"{config.tourn_size}")

    min_initial_tree_sld = ft.Slider(min=1, max=8, value=config.min_initial_tree_depth, divisions=7, width=300, on_change=slider_change, on_change_end=slider_change)
    min_initial_tree_label = ft.Text(f"{config.min_initial_tree_depth}")
    max_initial_tree_sld = ft.Slider(min=2, max=9, value=config.max_initial_tree_depth, divisions=7, width=300, on_change=slider_change, on_change_end=slider_change)
    max_initial_tree_label = ft.Text(f"{config.max_initial_tree_depth}")
    max_tree_depth_sld = ft.Slider(min=10, max=30, value=config.max_tree_depth, divisions=20, width=300, on_change=slider_change, on_change_end=slider_change)
    max_tree_depth_label = ft.Text(f"{config.max_tree_depth}")

    plot_adjust_swt = ft.Switch(value=config.plot_adjust, on_change=change_plot_adjust_switch)
    inter_gen_pause_label = ft.Text(f"{config.inter_gen_pause}")
    inter_gen_pause_sld = ft.Slider(min=0.0, max=5.0, value=config.inter_gen_pause, divisions=10, width=300, on_change=slider_change, on_change_end=slider_change)

    selection_method_dropdown = ft.Dropdown(
        content_padding=10,
        value=config.selection_method,
        on_change=changed_selection_method,
        options=[
            ft.dropdown.Option("Tournament"),
            ft.dropdown.Option("Random"),
            ft.dropdown.Option("Best"),
            ft.dropdown.Option("Worst"),
        ],
    )

    initialization_method_dropdown = ft.Dropdown(
        content_padding=10,
        value=config.initialization,
        on_change=changed_initialization_method,
        options=[
            ft.dropdown.Option("Ramped Half-n-Half"),
            ft.dropdown.Option("Full"),
            ft.dropdown.Option("Grow"),
        ],
    )

    dlg_title = ft.Text("")
    dlg_message = ft.Text("")
    dlg_modal = ft.AlertDialog(
        modal=True,
        title=ft.Row([ft.Icon(name=ft.icons.WARNING_AMBER, size=40), dlg_title]),
        content=dlg_message,
        actions=[
            ft.TextButton("Close", on_click=close_dlg),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    t = ft.Tabs(
        selected_index=0,
        animation_duration=0,
        tabs=[
            ft.Tab(
                text="Run",
                icon=ft.icons.PLAY_ARROW,
                content=ft.Row(
                    [                        
                        ft.Column(
                            [
                                ft.Container(height=2),
                                ft.Text("Benchmark Problem:"),
                                ft.Container(
                                    content=target_function_dropdown,
                                    margin = ft.margin.symmetric(horizontal=22),
                                ),
                                training_set_size_label,
                                training_set_size_sld,
                                noise_label,
                                noise_sld,
                                validation_set_size_label,
                                validation_set_size_sld,
                                generations_label,
                                generations_sld,
                                population_size_label,
                                population_size_sld,
                                crossover_probability_label,
                                crossover_probability_sld,
                                mutation_probability_label,
                                mutation_probability_sld,
                                ft.Container(
                                    content=ft.Row(
                                        [
                                            play_btn, 
                                            stop_btn, 
                                        ],
                                        alignment=ft.MainAxisAlignment.CENTER,
                                    ),
                                    margin = ft.margin.only(22, 15, 22, 10),
                                ),
                            ],
                            spacing=10,
                            expand=False,
                            width=400,
                        ),
                        ft.Column(
                            [
                                ft.Container(height=2),
                                ft.Row(
                                        [
                                            ft.Text("Training and Validation Error:", width=475),
                                            ft.Text("Best Individual vs. Training Points:"),
                                        ]
                                ),
                                ft.Container(
                                    content=MatplotlibChart(config.main_plots, expand=True),
                                    margin = ft.margin.only(-11, 0, -11, 0),      
                                ),
                                current_best_individual_label,
                                best_solution_textfield,
                                run_progress_bar,
                            ],
                            spacing=10,
                            expand=True,
                        ),
                    ],
                    spacing=10,
                    expand=True,
                ),
            ),
            ft.Tab(
                text="Function & Terminal Set",
                icon=ft.icons.FUNCTIONS,
                content=ft.DataTable(
                    column_spacing=100,
                    columns=[
                        ft.DataColumn(
                            ft.Text("Name", weight=ft.FontWeight.NORMAL),
                        ),
                        ft.DataColumn(
                            ft.Text("Type", weight=ft.FontWeight.NORMAL),
                        ),
                        ft.DataColumn(
                            ft.Text("Description", weight=ft.FontWeight.NORMAL),
                        ),
                        ft.DataColumn(
                            ft.Text("Selected", weight=ft.FontWeight.NORMAL),
                        ),
                    ],
                    rows=[
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("add(a, b)")), 
                                ft.DataCell(ft.Text("Function")), 
                                ft.DataCell(ft.Container(content=ft.Text("Adds two numbers."), width=750)),
                                ft.DataCell(add_func_swt)
                            ]
                        ),
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("sub(a, b)")), 
                                ft.DataCell(ft.Text("Function")), 
                                ft.DataCell(ft.Text("Subtracts two numbers.")),
                                ft.DataCell(sub_func_swt)
                            ]
                        ),
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("mul(a, b)")), 
                                ft.DataCell(ft.Text("Function")), 
                                ft.DataCell(ft.Text("Multiplies two numbers.")),
                                ft.DataCell(mul_func_swt)
                            ]
                        ),    
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("protectedDiv(a, b)")), 
                                ft.DataCell(ft.Text("Function")), 
                                ft.DataCell(ft.Text("Protected division of two numbers. If b equals 0 then 1 is returned.")),
                                ft.DataCell(protectedDiv_func_swt)
                            ]
                        ), 
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("analyticQuotient(a, b)")), 
                                ft.DataCell(ft.Text("Function")), 
                                ft.DataCell(ft.Text("Divides two numbers using the formula: a / sqrt(1 + b**2).")),
                                ft.DataCell(analyticQuotient_func_swt)
                            ]
                        ), 
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("neg(a)")), 
                                ft.DataCell(ft.Text("Function")), 
                                ft.DataCell(ft.Text("Negates the value of a.")),
                                ft.DataCell(neg_func_swt)
                            ]
                        ), 
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("sin(a)")), 
                                ft.DataCell(ft.Text("Function")), 
                                ft.DataCell(ft.Text("Sine of a.")),
                                ft.DataCell(sin_func_swt)
                            ]
                        ), 
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("cos(a)")), 
                                ft.DataCell(ft.Text("Function")), 
                                ft.DataCell(ft.Text("Cosine of a.")),
                                ft.DataCell(cos_func_swt)
                            ]
                        ), 
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("ERC Int [-1, 1]")), 
                                ft.DataCell(ft.Text("Terminal")), 
                                ft.DataCell(ft.Text("Ephemeral Random Constant of type integer from the interval [-1, 1].")),
                                ft.DataCell(int101_erc_swt)
                            ]
                        ), 
                    ],
                ),
            ),
            ft.Tab(
                text="Additional Parameters",
                icon=ft.icons.SETTINGS_OUTLINED,
                content=ft.DataTable(
                    column_spacing=100,
                    columns=[
                        ft.DataColumn(
                            ft.Text("Name", weight=ft.FontWeight.NORMAL),
                        ),
                        ft.DataColumn(
                            ft.Text("Type", weight=ft.FontWeight.NORMAL),
                        ),
                        ft.DataColumn(
                            ft.Text("Description", weight=ft.FontWeight.NORMAL),
                        ),
                        ft.DataColumn(
                            ft.Text("Input", weight=ft.FontWeight.NORMAL),
                        ),
                    ],
                    rows=[
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("Selection Method")), 
                                ft.DataCell(ft.Text("GP")), 
                                ft.DataCell(ft.Container(content=ft.Text("Method for the selection of individuals for the next generation."))),
                                ft.DataCell(
                                    ft.Container(
                                        content=selection_method_dropdown,
                                        width = 300,
                                        margin = ft.margin.symmetric(horizontal=0),
                                    ),
                                )
                            ]
                        ),
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("Tournament Size")), 
                                ft.DataCell(ft.Text("GP")), 
                                ft.DataCell(ft.Container(content=ft.Text("Size of the used tournament (if tournament selection is used)."))),
                                ft.DataCell(
                                    ft.Container(
                                        content=ft.Row(
                                            [
                                                tourn_size_sld, 
                                                tourn_size_label
                                            ],
                                            alignment=ft.MainAxisAlignment.CENTER,
                                        ),
                                        margin=ft.Margin(left=-37, top=0, right=0, bottom=0)
                                    ),
                                )
                            ]
                        ),
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("Initialization Method")), 
                                ft.DataCell(ft.Text("GP")), 
                                ft.DataCell(ft.Container(content=ft.Text("Method used for the initialization of the individuals in the first generation."))),
                                ft.DataCell(
                                    ft.Container(
                                        content=initialization_method_dropdown,
                                        width = 300,
                                        margin = ft.margin.symmetric(horizontal=0),
                                    ),
                                )
                            ]
                        ),
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("Minimum Initial Tree Depth")), 
                                ft.DataCell(ft.Text("GP")), 
                                ft.DataCell(ft.Container(content=ft.Text("Minimum tree depth of the individuals in the first generation"))),
                                ft.DataCell(
                                    ft.Container(
                                        content=ft.Row(
                                            [
                                                min_initial_tree_sld, 
                                                min_initial_tree_label
                                            ],
                                            alignment=ft.MainAxisAlignment.CENTER,
                                        ),
                                        margin=ft.Margin(left=-37, top=0, right=0, bottom=0)
                                    ),
                                )
                            ]
                        ),
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("Maximum Initial Tree Depth")), 
                                ft.DataCell(ft.Text("GP")), 
                                ft.DataCell(ft.Container(content=ft.Text("Maximum tree depth of the individuals in the first generation"))),
                                ft.DataCell(
                                    ft.Container(
                                        content=ft.Row(
                                            [
                                                max_initial_tree_sld, 
                                                max_initial_tree_label
                                            ],
                                            alignment=ft.MainAxisAlignment.CENTER,
                                        ),
                                        margin=ft.Margin(left=-37, top=0, right=0, bottom=0)
                                    ),
                                )
                            ]
                        ),
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("Maximum Tree Depth")), 
                                ft.DataCell(ft.Text("GP")), 
                                ft.DataCell(ft.Container(content=ft.Text("The maximum allowed tree depth."))),
                                ft.DataCell(
                                    ft.Container(
                                        content=ft.Row(
                                            [
                                                max_tree_depth_sld, 
                                                max_tree_depth_label
                                            ],
                                            alignment=ft.MainAxisAlignment.CENTER,
                                        ),
                                        margin=ft.Margin(left=-30, top=0, right=0, bottom=0)
                                    ),
                                )
                            ]
                        ),
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("Plot Adjustment")), 
                                ft.DataCell(ft.Text("APP")), 
                                ft.DataCell(ft.Container(content=ft.Text("Method used for adjusting the Y axis of the plots."))),
                                ft.DataCell(
                                    ft.Container(
                                        content=plot_adjust_swt,
                                        margin=ft.Margin(left=-5, top=0, right=0, bottom=0)
                                    ),
                                ) 
                            ]
                        ),
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text("Inter-Generation Pause")),    
                                ft.DataCell(ft.Text("APP")), 
                                ft.DataCell(ft.Container(content=ft.Text("The minimum pause time between displaying two generations (in seconds)."))),
                                ft.DataCell(
                                    ft.Container(
                                        content=ft.Row(
                                            [
                                                inter_gen_pause_sld, 
                                                inter_gen_pause_label
                                            ],
                                            alignment=ft.MainAxisAlignment.CENTER,
                                        ),
                                        margin=ft.Margin(left=-27, top=0, right=0, bottom=0)
                                    ),
                                ) 
                            ]
                        ),
                    ],
                ),
            ),
            ft.Tab(
                text="Info",
                icon=ft.icons.INFO_OUTLINED,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text(
                            "GPExplore v0.1",
                            size=28,
                            weight=ft.FontWeight.BOLD,
                            text_align=ft.TextAlign.CENTER,
                        ),
                        ft.Row([
                            ft.Text(
                                "GitHub: ",
                                size=16,
                                text_align=ft.TextAlign.START,
                            ),
                            ft.Container(
                                content=ft.TextButton(
                                    text="https://github.com/domsob/gpexplore", 
                                    url="https://github.com/domsob/gpexplore", 
                                    style=ft.ButtonStyle(
                                        mouse_cursor=ft.MouseCursor.CLICK
                                    ),
                                ),
                                margin=ft.Margin(left=-12, top=0, right=0, bottom=0),
                            ),
                        ]),
                        ft.Text(
                            "Contributor(s): Dominik Sobania",
                            size=16,
                            text_align=ft.TextAlign.START,
                        ),
                        ft.TextField(
                            label="License",
                            value=config.license_text,
                            read_only=True,
                            multiline=True,
                            height=license_height,
                        )
                    ],
                    spacing=15,
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                ),
                margin=ft.Margin(left=5, top=5, right=5, bottom=20),
            ),),
        ],
        expand=1,
    )
    page.add(t)

ft.app(target=main)
#ft.app(target=main, view=ft.AppView.WEB_BROWSER)