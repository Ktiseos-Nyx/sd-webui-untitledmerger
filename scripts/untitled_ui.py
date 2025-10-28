import gradio as gr
import os
import re
import functools
import json
import shutil
import torch
import safetensors
import safetensors.torch
from modules import sd_models,script_callbacks,scripts,shared,ui_components,paths,sd_samplers,ui,call_queue
from modules.ui_common import create_output_panel,plaintext_to_html, create_refresh_button
# from modules.ui import create_sampler_and_steps_selection
from scripts.untitled import merger,misc_util
from scripts.untitled.operators import weights_cache
from scripts.untitled import lora_merge
import scripts.untitled.common as cmn

extension_path = scripts.basedir()

ext2abs = lambda *x: os.path.join(extension_path,*x)

sd_checkpoints_path = os.path.join(paths.models_path,'checklora')

options_filename = ext2abs('scripts','untitled','options.json')

custom_sliders_examples = ext2abs('scripts','untitled','sliders_examples.json')
custom_sliders_presets = ext2abs('scripts','untitled','custom_sliders_presets.json')
loaded_slider_presets = None

with open(ext2abs('scripts','examplemerge.yaml'), 'r') as file:
    EXAMPLE = file.read()

model_a_keys = []


class Progress:
    def __init__(self):
        self.ui_report = []
        self.merge_keys = 0

    def __call__(self,message, v=None, popup = False, report=False):
        if v:
            message = ' - '+ message + ' ' * (25-len(message)) + ': ' + str(v)

        if report:
            self.ui_report.append(message)

        if popup:
            gr.Info(message)

        print(message)

    def interrupt(self,message,popup=True):
        message = 'Merge interrupted:\t'+message

        if popup:
            gr.Warning(message)

        self.ui_report = [message]
        raise merger.MergeInterruptedError

    def get_report(self):
        return '\n'.join(self.ui_report)


class Options:
    def __init__(self,filename):
        self.filename = filename
        try:
            with open(filename,'r') as file:
                self.options = json.load(file)
        except FileNotFoundError:
            self.options = dict()

    def create_option(self,key,component,component_kwargs,default):
        value = self.options.get(key) or default

        opt_component = component(value = value,**component_kwargs)
        opt_component.do_not_save_to_config = True
        self.options[key] = value
        def opt_event(value): self.options[key] = value
        opt_component.change(fn=opt_event, inputs=opt_component)
        return opt_component

    def __getitem__(self,key):
        return self.options.get(key)

    def save(self):
        with open(self.filename,'w') as file:
            json.dump(self.options,file,indent=4)
        gr.Info('Options saved')

cmn.opts = Options(options_filename)


def on_ui_tabs():
    with gr.Blocks() as cmn.blocks:
        with gr.Tab("Merge"):
            dummy_component = gr.Textbox(visible=False,interactive=True)
            with ui_components.ResizeHandleRow():
                with gr.Column():
                    status = gr.Textbox(max_lines=4,lines=4,show_label=False,info="",interactive=False,render=False)
                    #### MODEL SELECTION
                    with gr.Row():
                        slider_scale = 8
                        with gr.Column(variant='compact',min_width=150,scale=slider_scale):
                            with gr.Row():
                                model_a = gr.Dropdown(get_checkpoints_list('Alphabetical'), label="model_a [Primary]",scale=slider_scale)
                                swap_models_AB = gr.Button(value='⇆', elem_classes=["tool"],scale=1)
                            model_a_info = gr.HTML(plaintext_to_html('None | None',classname='untitled_sd_version'))
                            model_a.change(fn=checkpoint_changed,inputs=model_a,outputs=model_a_info).then(fn=update_model_a_keys, inputs=model_a)
                
                        with gr.Column(variant='compact',min_width=150,scale=slider_scale):
                            with gr.Row():
                                model_b = gr.Dropdown(get_checkpoints_list('Alphabetical'), label="model_b [Secondary]",scale=slider_scale)
                                swap_models_BC = gr.Button(value='⇆', elem_classes=["tool"],scale=1)
                            model_b_info = gr.HTML(plaintext_to_html('None | None',classname='untitled_sd_version'))
                            model_b.change(fn=checkpoint_changed,inputs=model_b,outputs=model_b_info)
                
                        with gr.Column(variant='compact',min_width=150,scale=slider_scale):
                            with gr.Row():
                                model_c = gr.Dropdown(get_checkpoints_list('Alphabetical'), label="model_c [Tertiary]",scale=slider_scale)
                                swap_models_CD = gr.Button(value='⇆', elem_classes=["tool"],scale=1)
                            model_c_info = gr.HTML(plaintext_to_html('None | None',classname='untitled_sd_version'))
                            model_c.change(fn=checkpoint_changed,inputs=model_c,outputs=model_c_info)
                
                        with gr.Column(variant='compact',min_width=150,scale=slider_scale):
                            with gr.Row():
                                model_d = gr.Dropdown(get_checkpoints_list('Alphabetical'), label="model_d [Supplementary]",scale=slider_scale)
                                refresh_button = gr.Button(value='🔄', elem_classes=["tool"],scale=1)
                            model_d_info = gr.HTML(plaintext_to_html('None | None',classname='untitled_sd_version'))
                            model_d.change(fn=checkpoint_changed,inputs=model_d,outputs=model_d_info)
                
                        checkpoint_sort = gr.Dropdown(min_width=60,scale=1,visible=True,choices=['Alphabetical','Newest first'],value='Alphabetical',label='Sort')
                
                        def swapvalues(x,y): return gr.update(value=y), gr.update(value=x)
                        swap_models_AB.click(fn=swapvalues,inputs=[model_a,model_b],outputs=[model_a,model_b])
                        swap_models_BC.click(fn=swapvalues,inputs=[model_b,model_c],outputs=[model_b,model_c])
                        swap_models_CD.click(fn=swapvalues,inputs=[model_c,model_d],outputs=[model_c,model_d])
                        refresh_button.click(fn=refresh_models,inputs=checkpoint_sort, outputs=[model_a,model_b,model_c,model_d])
                        checkpoint_sort.change(fn=refresh_models,inputs=checkpoint_sort,outputs=[model_a,model_b,model_c,model_d])


                    #### MODE SELECTION
                    with gr.Row():
                        merge_mode_selector = gr.Radio(label='Merge Mode (formula structure):',choices=list(merger.mergemode_selection.keys()),value=list(merger.mergemode_selection.keys())[0],scale=3)
                    merge_mode_desc = gr.Textbox(label="Merge Mode Description", value=merger.mergemode_selection[list(merger.mergemode_selection.keys())[0]].description, interactive=False, lines=2)

                    with gr.Row():
                        calc_mode_selector = gr.Radio(label='Calculation Mode (how to execute):',choices=list(merger.calcmode_selection.keys()),value=list(merger.calcmode_selection.keys())[0],scale=3)
                    calc_mode_desc = gr.Textbox(label="Calculation Mode Description", value=merger.calcmode_selection[list(merger.calcmode_selection.keys())[0]].description, interactive=False, lines=2)


                    ##### MAIN SLIDERS
                    with gr.Row(equal_height=True):
                        alpha = gr.Slider(minimum=-1,step=0.01,maximum=2,label="slider_a [α] (alpha)",info='model_a - model_b',value=0.5,elem_classes=['main_sliders'])
                        beta = gr.Slider(minimum=-1,step=0.01,maximum=2,label="slider_b [β] (beta)",info='-',value=0.5,elem_classes=['main_sliders'])
                        gamma = gr.Slider(minimum=-1,step=0.01,maximum=2,label="slider_c [γ] (gamma)",info='-',value=0.25,elem_classes=['main_sliders'])
                        delta = gr.Slider(minimum=-1,step=0.01,maximum=2,label="slider_d [δ] (delta)",info='-',value=0.25,elem_classes=['main_sliders'])

                    merge_mode_selector.change(fn=mode_changed, inputs=[merge_mode_selector, calc_mode_selector], outputs=[merge_mode_desc,calc_mode_desc,alpha,beta,gamma,delta],show_progress='hidden')
                    calc_mode_selector.change(fn=mode_changed, inputs=[merge_mode_selector, calc_mode_selector], outputs=[merge_mode_desc,calc_mode_desc,alpha,beta,gamma,delta],show_progress='hidden')
            
            
                    ### SAVING
                    with gr.Row(equal_height=True):
                        with gr.Column(variant='panel'):
                            save_name = gr.Textbox(max_lines=1,label='Save checkpoint as:',lines=1,placeholder='Enter name...',scale=2)
                            with gr.Row():
                                save_settings = gr.CheckboxGroup(label = " ",choices=["Autosave","Overwrite","fp16","bf16"],value=['fp16'],interactive=True,scale=2,min_width=100)
                                save_loaded = gr.Button(value='Save loaded checkpoint',size='sm',scale=1)
                                save_loaded.click(fn=misc_util.save_loaded_model, inputs=[save_name,save_settings],outputs=status).then(fn=refresh_models, inputs=checkpoint_sort,outputs=[model_a,model_b,model_c,model_d])
            
                    #### MERGE BUTTONS
                        with gr.Column():
                            merge_button = gr.Button(value='Merge',variant='primary')
                            # merge_and_gen_button = gr.Button(value='Merge & Gen',variant='primary')
                            with gr.Row():
                                empty_cache_button = gr.Button(value='Empty Cache')
                                empty_cache_button.click(fn=merger.clear_cache,outputs=status)
            
                                stop_button = gr.Button(value='Stop')
                                def stopfunc(): cmn.stop = True;shared.state.interrupt()
                                stop_button.click(fn=stopfunc)
                            #### SNEED
                            with gr.Row():
                                merge_seed = gr.Number(label='Merge Seed', value=99,  min_width=100, precision=0,scale=1)
                                merge_random_seed = ui_components.ToolButton(ui.random_symbol, tooltip="Set seed to -1, which will cause a new random number to be used every time")
                                merge_random_seed.click(fn=lambda:-1, outputs=merge_seed)
                                merge_reuse_seed = ui_components.ToolButton(ui.reuse_symbol, tooltip="Reuse seed from last generation, mostly useful if it was randomized")
                                merge_reuse_seed.click(fn=lambda:cmn.last_merge_seed, outputs=merge_seed)
            
                    ### INCLUDE EXCLUDE
                    with gr.Accordion(label='Include/Exclude/Discard',open=False):
                        with gr.Row():
                            with gr.Column():
                                clude = gr.Textbox(max_lines=4,label='Include/Exclude:',info='Entered targets will remain as model_a when set to \'Exclude\', and will be the only ones to be merged if set to \'Include\'. Separate with withspace.',value='clip',lines=4,scale=4)
                                clude_mode = gr.Radio(label="",info="",choices=["Exclude",("Include exclusively",'include')],value='Exclude',min_width=300,scale=1)
                            discard = gr.Textbox(max_lines=5,label='Discard:',info="Targets will be removed from the model, only applies to autosaved models. Separate with whitespace.",value='model_ema',lines=5,scale=1)
            
                    ### CUSTOM SLIDERS
                    with ui_components.InputAccordion(False, label='Custom sliders') as enable_sliders:
            
                        with gr.Accordion(label = 'Presets'):
                            with gr.Row(variant='compact'):
                                sliders_preset_dropdown = gr.Dropdown(label='Preset Name',allow_custom_value=True,choices=get_slider_presets(),value='blocks',scale=4)
            
                                slider_refresh_button = gr.Button(value='🔄', elem_classes=["tool"],scale=1,min_width=40)
                                slider_refresh_button.click(fn=lambda:gr.update(choices=get_slider_presets()),outputs=sliders_preset_dropdown)
            
                                sliders_preset_load = gr.Button(variant='secondary',value='Load presets',scale=2)
                                sliders_preset_save = gr.Button(variant='secondary',value='Save sliders as preset',scale=2)
            
                            with open(custom_sliders_examples,'r') as file:
                                presets = json.load(file)
                            slid_defaults = iter(presets['blocks'])
            
                            slider_slider = gr.Slider(step=2,maximum=26,value=slid_defaults.__next__(),label='Enabled Sliders')
            
                        custom_sliders = []
                        with gr.Row():
                            for w in [6,1,6]:
                                with gr.Column(scale=w,min_width=0):
                                    if w>1:
                                        for i in range(13):
                                            with gr.Row(variant='compact'):
                                                custom_sliders.append(gr.Textbox(show_label=False,visible=True,value=slid_defaults.__next__(),placeholder='target',min_width=100,scale=1,lines=1,max_lines=1))
                                                custom_sliders.append(gr.Slider(show_label=False,value=slid_defaults.__next__(),scale=6,minimum=0,maximum=1,step=0.01))
            
                        def show_sliders(n):
                            n = int(n/2)
                            update_column = [gr.update(visible=True), gr.update(visible=True)]*n + [gr.update(visible=False), gr.update(visible=False)]*(13-n)
                            return update_column * 2
            
                        slider_slider.change(fn=show_sliders,inputs=slider_slider,outputs=custom_sliders,show_progress='hidden')
                        slider_slider.release(fn=show_sliders,inputs=slider_slider,outputs=custom_sliders,show_progress='hidden')
            
                        sliders_preset_save.click(fn=save_custom_sliders,inputs=[sliders_preset_dropdown,slider_slider,*custom_sliders])
                        sliders_preset_load.click(fn=load_slider_preset,inputs=[sliders_preset_dropdown],outputs=[slider_slider,*custom_sliders])
            
                    ### ADJUST
                    with gr.Accordion("Supermerger Adjust", open=False) as acc_ad:
                        with gr.Row(variant="compact"):
                            finetune = gr.Textbox(label="Adjust", show_label=False, info="Adjust IN,OUT,OUT2,Contrast,Brightness,COL1,COL2,COL3", visible=True, value="", lines=1)
                            finetune_write = gr.Button(value="↑", elem_classes=["tool"])
                            finetune_read = gr.Button(value="↓", elem_classes=["tool"])
                            finetune_reset = gr.Button(value="\U0001f5d1\ufe0f", elem_classes=["tool"])
                        with gr.Row(variant="compact"):
                            with gr.Column(scale=1, min_width=100):
                                detail1 = gr.Slider(label="IN", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                            with gr.Column(scale=1, min_width=100):
                                detail2 = gr.Slider(label="OUT", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                            with gr.Column(scale=1, min_width=100):
                                detail3 = gr.Slider(label="OUT2", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                        with gr.Row(variant="compact"):
                            with gr.Column(scale=1, min_width=100):
                                contrast = gr.Slider(label="Contrast", minimum=-10, maximum=10, step=0.01, value=0, info="Contrast/Detail")
                            with gr.Column(scale=1, min_width=100):
                                bri = gr.Slider(label="Brightness", minimum=-10, maximum=10, step=0.01, value=0, info="Dark(Minius)-Bright(Plus)")
                        with gr.Row(variant="compact"):
                            with gr.Column(scale=1, min_width=100):
                                col1 = gr.Slider(label="Cyan-Red", minimum=-10, maximum=10, step=0.01, value=0, info="Cyan(Minius)-Red(Plus)")
                            with gr.Column(scale=1, min_width=100):
                                col2 = gr.Slider(label="Magenta-Green", minimum=-10, maximum=10, step=0.01, value=0, info="Magenta(Minius)-Green(Plus)")
                            with gr.Column(scale=1, min_width=100):
                                col3 = gr.Slider(label="Yellow-Blue", minimum=-10, maximum=10, step=0.01, value=0, info="Yellow(Minius)-Blue(Plus)")
            
                            finetune.change(fn=lambda x:gr.update(label = f"Supermerger Adjust : {x}"if x != "" and x !="0,0,0,0,0,0,0,0" else "Supermerger Adjust"),inputs=[finetune],outputs = [acc_ad])
            
                        def finetune_update(finetune, detail1, detail2, detail3, contrast, bri, col1, col2, col3):
                            arr = [detail1, detail2, detail3, contrast, bri, col1, col2, col3]
                            tmp = ",".join(map(lambda x: str(int(x)) if x == 0.0 else str(x), arr))
                            if finetune != tmp:
                                return gr.update(value=tmp)
                            return gr.update()
            
                        def finetune_reader(finetune):
                            try:
                                tmp = [float(t) for t in finetune.split(",") if t]
                                assert len(tmp) == 8, f"expected 8 values, received {len(tmp)}."
                            except ValueError as err: gr.Warning(str(err))
                            except AssertionError as err: gr.Warning(str(err))
                            else: return [gr.update(value=x) for x in tmp]
                            return [gr.update()]*8
            
                        # update finetune
                        finetunes = [detail1, detail2, detail3, contrast, bri, col1, col2, col3]
                        finetune_reset.click(fn=lambda: [gr.update(value="")]+[gr.update(value=0.0)]*8, inputs=[], outputs=[finetune, *finetunes])
                        finetune_read.click(fn=finetune_reader, inputs=[finetune], outputs=[*finetunes])
                        finetune_write.click(fn=finetune_update, inputs=[finetune, *finetunes], outputs=[finetune])
                        detail1.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                        detail2.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                        detail3.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                        contrast.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                        bri.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                        col1.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                        col2.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                        col3.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
            
                    ###OPTIONS####
                    with gr.Accordion(label='Options',open=False):
                        save_options_button = gr.Button(value = 'Save',variant='primary')
                        save_options_button.click(fn=cmn.opts.save)
                        cmn.opts.create_option('trash_model',
                                            gr.Radio,
                                            {'choices':['Disable','Enable for SDXL','Enable'],
                                                'label':'Clear loaded SD models from memory at the start of merge:',
                                                'info':'Saves some memory but increases loading times'},
                                                default='Enable for SDXL')
            
                        cmn.opts.create_option('device',
                                            gr.Radio,
                                            {'choices':['cuda/float16', 'cuda/float32', 'cpu/float32'],
                                                'label':'Preferred device/dtype for merging:'},
                                                default='cuda/float16')
            
                        cmn.opts.create_option('threads',
                                            gr.Slider,
                                            {'step':2,
                                                'minimum':2,
                                                'maximum':20,
                                                'label':'Worker thread count:',
                                                'info':'Relevant for both cuda and CPU merging. Using too many threads can harm performance. Your core-count +-2 is a good guideline.'},
                                                default=8)
            
                        cache_size_slider = cmn.opts.create_option('cache_size',
                                            gr.Slider,
                                            {'step':64,
                                                'minimum':0,
                                                'maximum':16384,
                                                'label':'Cache size (MB):',
                                                'info':'Stores the result of intermediate calculations, such as the difference between B and C in add-difference before its multiplied and added to A.'},
                                                default=4096)
            
                    cache_size_slider.release(fn=lambda x: weights_cache.__init__(x),inputs=cache_size_slider)
                    weights_cache.__init__(cmn.opts['cache_size'])
            
            
                gen_elem_id = 'untitled_merger'
            
                #model_prefix = gr.Textbox(max_lines=1,lines=1,label='Prefix checkpoint filenames', info='Use / to save checkpoints to a subfolder.',placeholder='folder/merge_')
            
                with gr.Column():
                    status.render()
                    with gr.Accordion('Weight editor'):
                        weight_editor = gr.Code(value=EXAMPLE,lines=20,language='yaml',label='')
                    #with gr.Tab(label='Image gen'):
                    #    with gr.Column(variant='panel'):
                    #        try:
                    #            output_panel = create_output_panel('untitled_merger', shared.opts.outdir_txt2img_samples)
                    #            output_gallery, output_html_log = output_panel.gallery, output_panel.html_log
                    #        except: #for compatibiltiy with webui 1.7.0 and older
                    #            output_gallery, _, _, output_html_log = create_output_panel('untitled_merger', shared.opts.outdir_txt2img_samples)
                    #
                    #        with gr.Row(equal_height=False):
                    #            with gr.Accordion(label='Generation info',open=False):
                    #                infotext = gr.HTML(elem_id=f'html_info_{gen_elem_id}', elem_classes="infotext",scale=3)
                    #            gen_button = gr.Button(value='Gen',variant='primary',scale=1)
                    #
                    #    with gr.Row(elem_id=f"{gen_elem_id}_prompt_container", elem_classes=["prompt-container-compact"],equal_height=True):
                    #            promptbox = gr.Textbox(label="Prompt", elem_id=f"{gen_elem_id}_prompt", show_label=False, lines=3, placeholder="Prompt", elem_classes=["prompt"])
                    #            negative_promptbox = gr.Textbox(label="Negative prompt", elem_id=f"{gen_elem_id}_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt", elem_classes=["prompt"])
                    #    steps, sampler_name = create_sampler_and_steps_selection(sd_samplers.visible_sampler_names(), gen_elem_id)
                    #
                    #
                    #    with ui_components.FormRow():
                    #        with gr.Column(elem_id=f"{gen_elem_id}_column_size", scale=4):
                    #            width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id=f"{gen_elem_id}_width")
                    #            height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id=f"{gen_elem_id}_height")
                    #
                    #        with gr.Column(elem_id=f"{gen_elem_id}_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                    #            res_switch_btn = gr.Button(value='⇅', elem_id=f"{gen_elem_id}_res_switch_btn", tooltip="Switch width/height", elem_classes=["tool"])
                    #            res_switch_btn.click(fn=swapvalues, inputs=[width,height], outputs=[width,height])
                    #
                    #        with gr.Column(elem_id=f"{gen_elem_id}_column_batch"):
                    #                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id=f"{gen_elem_id}_batch_count")
                    #                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id=f"{gen_elem_id}_batch_size")
                    #
                    #    with gr.Row():
                    #        cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id=f"{gen_elem_id}_cfg_scale")
                    #
                    #    with gr.Row():
                    #        seed = gr.Number(label='Seed', value=99, elem_id=gen_elem_id+" seed", min_width=100, precision=0)
                    #
                    #        random_seed = ui_components.ToolButton(ui.random_symbol, elem_id=gen_elem_id+" random_seed", tooltip="Set seed to -1, which will cause a new random number to be used every time")
                    #        random_seed.click(fn=lambda:-1, outputs=seed)
                    #        reuse_seed = ui_components.ToolButton(ui.reuse_symbol, elem_id=gen_elem_id+" reuse_seed", tooltip="Reuse seed from last generation, mostly useful if it was randomized")
                    #        reuse_seed.click(fn=lambda:cmn.last_seed, outputs=seed)
                    #
                    #
                    #    with ui_components.InputAccordion(False, label="Hires. fix", elem_id=f"{gen_elem_id}_hr") as enable_hr:
                    #        with enable_hr.extra():
                    #            hr_final_resolution = ui_components.FormHTML(value="", elem_id=f"{gen_elem_id}_hr_finalres", label="Upscaled resolution", interactive=False, min_width=0)
                    #
                    #        with ui_components.FormRow(elem_id=f"{gen_elem_id}_hires_fix_row1", variant="compact"):
                    #            hr_upscaler = gr.Dropdown(label="Upscaler", elem_id=f"{gen_elem_id}_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                    #            hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id=f"{gen_elem_id}_hires_steps")
                    #            denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id=f"{gen_elem_id}_denoising_strength")
                    #
                    #        with ui_components.FormRow(elem_id=f"{gen_elem_id}_hires_fix_row2", variant="compact"):
                    #            hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id=f"{gen_elem_id}_hr_scale")
                    #            hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=0, elem_id=f"{gen_elem_id}_hr_resize_x")
                    #            hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=0, elem_id=f"{gen_elem_id}_hr_resize_y")
                    #
                    #        hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y]
                    #
                    #        for component in hr_resolution_preview_inputs:
                    #            event = component.release if isinstance(component, gr.Slider) else component.change
                    #
                    #            event(
                    #                fn=ui.calc_resolution_hires,
                    #                inputs=hr_resolution_preview_inputs,
                    #                outputs=[hr_final_resolution],
                    #                show_progress=False,
                    #            )
            
                    with gr.Accordion('Model keys'):
                        target_tester = gr.Textbox(max_lines=1,label="Checks model_a keys using simple expression.",info="'*' is used as wildcard. Start expression with 'cond*' for clip. 'c*embedders.0*' for small clip. 'c*embedders.1*' for big clip. 'model.*' for unet and 'model_ema*' for ema unet",interactive=True,placeholder='model.*out*4*tran*norm*weight')
                        target_tester_display = gr.Textbox(max_lines=40,lines=40,label="Targeted keys:",info="",interactive=False)
                        target_tester.change(fn=test_regex,inputs=[target_tester],outputs=target_tester_display,show_progress='minimal')
            
                merge_args = [
                    finetune,
                    merge_mode_selector,
                    calc_mode_selector,
                    model_a,
                    model_b,
                    model_c,
                    model_d,
                    alpha,
                    beta,
                    gamma,
                    delta,
                    weight_editor,
                    discard,
                    clude,
                    clude_mode,
                    merge_seed,
                    enable_sliders,
                    slider_slider,
                    *custom_sliders
                    ]
            
                #gen_args = [
                #    dummy_component,
                #    promptbox,
                #    negative_promptbox,
                #    steps,
                #    sampler_name,
                #    width,
                #    height,
                #    batch_count,
                #    batch_size,
                #    cfg_scale,
                #    seed,
                #    enable_hr,
                #    hr_upscaler,
                #    hr_second_pass_steps,
                #    denoising_strength,
                #    hr_scale,
                #    hr_resize_x,
                #    hr_resize_y
                #]
            
                merge_button.click(fn=start_merge,inputs=[save_name,save_settings,*merge_args],outputs=status)
            
                def merge_interrupted(func):
                    @functools.wraps(func)
                    def inner(*args):
                        if not cmn.interrupted:
                            return func(*args)
                        else:
                            return gr.update(),gr.update(),gr.update()
                    return inner
            
                # merge_and_gen_button.click(fn=start_merge,
                #                            inputs=[save_name,save_settings,*merge_args],
                #                            outputs=status).then(
                #                             fn=merge_interrupted(call_queue.wrap_gradio_gpu_call(misc_util.image_gen, extra_outputs=[None, '', ''])),
                #                             _js='submit_imagegen',
                #                             inputs=gen_args,
                #                             outputs=[output_gallery,infotext,output_html_log]
                # )
            
                # gen_button.click(fn=call_queue.wrap_gradio_gpu_call(misc_util.image_gen, extra_outputs=[None, '', '']),
                #                 _js='submit_imagegen',
                #                 inputs=gen_args,
                #                 outputs=[output_gallery,infotext,output_html_log])
        with gr.Tab("LoRA", elem_id="tab_lora"):
            gr.Markdown("## ⚠️ UNDER CONSTRUCTION\nLoRA merging is functional but experimental. Needs real-world testing and refinement. Use at your own risk!")
            lora_status = gr.Textbox(max_lines=4,lines=4,show_label=False,info="",interactive=False,render=False)

            with gr.Accordion("Merge LoRA(s) to Checkpoint", open=True):
                gr.Markdown("Bake one or multiple LoRAs into a checkpoint permanently")

                with gr.Row():
                    lora_to_ckpt_checkpoint = gr.Dropdown(get_checkpoints_list('Alphabetical'), label="Base Checkpoint", scale=4)
                    lora_refresh_ckpt = create_refresh_button(lora_to_ckpt_checkpoint, lambda: None, lambda: {'choices': get_checkpoints_list('Alphabetical')}, 'refresh_lora_ckpt')

                lora_ckpt_info = gr.HTML(plaintext_to_html('None | None',classname='untitled_sd_version'))

                with gr.Row():
                    lora_checkbox_group = gr.CheckboxGroup(label="Select LoRAs to merge (with dtype info)", choices=[], type="value", interactive=True)
                    lora_refresh_list = create_refresh_button(lora_checkbox_group, lambda: None, lambda: {'choices': get_lora_list()}, 'refresh_lora_list')

                with gr.Row():
                    lora_to_ckpt_strength = gr.Slider(minimum=-2, maximum=2, step=0.01, value=1.0, label="Global LoRA Strength", info="Apply this strength to all checked LoRAs (1.0 = normal)")
                    lora_to_ckpt_name = gr.Textbox(label="Output Name", placeholder="model_with_loras", info="Name for merged checkpoint")

                lora_to_ckpt_button = gr.Button("Merge LoRA(s) to Checkpoint", variant="primary")

            with gr.Accordion("Merge Multiple LoRAs", open=False):
                gr.Markdown("Combine 2-3 LoRA files into a single LoRA")
                with gr.Column():
                    with gr.Row():
                        lora_merge_lora1 = gr.Textbox(label="LoRA 1 Path", placeholder="/path/to/lora1.safetensors", scale=3)
                        lora_merge_weight1 = gr.Slider(minimum=0, maximum=2, step=0.01, value=0.5, label="Weight", scale=1)

                    with gr.Row():
                        lora_merge_lora2 = gr.Textbox(label="LoRA 2 Path", placeholder="/path/to/lora2.safetensors", scale=3)
                        lora_merge_weight2 = gr.Slider(minimum=0, maximum=2, step=0.01, value=0.5, label="Weight", scale=1)

                    with gr.Row():
                        lora_merge_lora3 = gr.Textbox(label="LoRA 3 Path (optional)", placeholder="/path/to/lora3.safetensors", scale=3)
                        lora_merge_weight3 = gr.Slider(minimum=0, maximum=2, step=0.01, value=0.0, label="Weight", scale=1)

                gr.Markdown("*Weights will be normalized to sum to 1.0*")
                with gr.Row():
                    lora_merge_name = gr.Textbox(label="Output Name", placeholder="merged_lora", info="Name for merged LoRA file")
                    lora_merge_button = gr.Button("Merge LoRAs", variant="primary")

            lora_status.render()

            # Wire up dtype detection
            lora_to_ckpt_checkpoint.change(fn=checkpoint_changed, inputs=lora_to_ckpt_checkpoint, outputs=lora_ckpt_info)

            # Wire up the LoRA merge buttons
            lora_to_ckpt_button.click(
                fn=merge_loras_to_checkpoint_ui,
                inputs=[lora_to_ckpt_checkpoint, lora_checkbox_group, lora_to_ckpt_name, lora_to_ckpt_strength],
                outputs=lora_status
            )

            lora_merge_button.click(
                fn=merge_loras_ui,
                inputs=[
                    lora_merge_lora1, lora_merge_weight1,
                    lora_merge_lora2, lora_merge_weight2,
                    lora_merge_lora3, lora_merge_weight3,
                    lora_merge_name
                ],
                outputs=lora_status
            )


    return [(cmn.blocks, "Untitled merger", "untitled_merger")]

script_callbacks.on_ui_tabs(on_ui_tabs)


def start_merge(*args):
    progress = Progress()

    try:
        merger.prepare_merge(progress, *args)
    except Exception as error:

        merger.clear_cache()
        if not shared.sd_model:
            sd_models.reload_model_weights(forced_reload=True)

        if not isinstance(error,merger.MergeInterruptedError):
            raise

    return progress.get_report()


def test_regex(input):
    regex = misc_util.target_to_regex(input)
    selected_keys = re.findall(regex,'\n'.join(model_a_keys),re.M)
    joined = '\n'.join(selected_keys)
    return  f'Matched keys: {len(selected_keys)}\n{joined}'


def update_model_a_keys(model_a):
    global model_a_keys
    path = sd_models.get_closet_checkpoint_match(model_a).filename
    with safetensors.torch.safe_open(path,framework='pt',device='cpu') as file:
        model_a_keys = file.keys()


def checkpoint_changed(name):
    if name == "":
        return plaintext_to_html('None | None',classname='untitled_sd_version')
    sdversion, dtype = misc_util.id_checkpoint(name)
    return plaintext_to_html(f"{sdversion} | {str(dtype).split('.')[1]}",classname='untitled_sd_version')


def lora_file_changed(lora_path):
    """Detect dtype of a LoRA file"""
    try:
        if not lora_path or not os.path.exists(lora_path):
            return plaintext_to_html('Invalid path',classname='untitled_sd_version')

        with safetensors.torch.safe_open(lora_path, framework='pt', device='cpu') as f:
            keys = list(f.keys())
            if len(keys) > 0:
                # Get dtype from first tensor
                tensor = f.get_tensor(keys[0])
                dtype_str = str(tensor.dtype).split('.')[1]
                num_keys = len(keys)
                return plaintext_to_html(f'LoRA | {dtype_str} | {num_keys} keys',classname='untitled_sd_version')
            else:
                return plaintext_to_html('LoRA | empty file',classname='untitled_sd_version')
    except Exception as e:
        return plaintext_to_html(f'Error: {str(e)}',classname='untitled_sd_version')


def mode_changed(mergemode_name, calcmode_name):
    """Handle changes to merge mode or calc mode selections"""
    mergemode = merger.mergemode_selection[mergemode_name]
    calcmode = merger.calcmode_selection[calcmode_name]

    # Use calc mode slider config if defined, otherwise use merge mode config
    # CalcModes like 'extract' have their own slider configs that override the merge mode
    if hasattr(calcmode, 'slid_a_config') and calcmode.slid_a_config != (-1, 2, 0.01):
        slider_a_config = calcmode.slid_a_config
        slider_a_info = calcmode.slid_a_info
        slider_b_config = calcmode.slid_b_config
        slider_b_info = calcmode.slid_b_info
        slider_c_config = calcmode.slid_c_config
        slider_c_info = calcmode.slid_c_info
        slider_d_config = calcmode.slid_d_config
        slider_d_info = calcmode.slid_d_info
    else:
        # Use merge mode's slider config
        slider_a_config = mergemode.slid_a_config
        slider_a_info = mergemode.slid_a_info
        slider_b_config = mergemode.slid_b_config
        slider_b_info = mergemode.slid_b_info
        slider_c_config = mergemode.slid_c_config
        slider_c_info = mergemode.slid_c_info
        slider_d_config = mergemode.slid_d_config
        slider_d_info = mergemode.slid_d_info

    slider_a_update = gr.update(minimum=slider_a_config[0], maximum=slider_a_config[1], step=slider_a_config[2], info=slider_a_info)
    slider_b_update = gr.update(minimum=slider_b_config[0], maximum=slider_b_config[1], step=slider_b_config[2], info=slider_b_info)
    slider_c_update = gr.update(minimum=slider_c_config[0], maximum=slider_c_config[1], step=slider_c_config[2], info=slider_c_info)
    slider_d_update = gr.update(minimum=slider_d_config[0], maximum=slider_d_config[1], step=slider_d_config[2], info=slider_d_info)

    # Return updates for description textboxes and all sliders
    merge_desc_update = gr.update(value=mergemode.description)
    calc_desc_update = gr.update(value=calcmode.description)

    return merge_desc_update, calc_desc_update, slider_a_update, slider_b_update, slider_c_update, slider_d_update


def get_checkpoints_list(sort):
    checkpoints_list = [x.title for x in sd_models.checkpoints_list.values() if x.is_safetensors]
    if sort == 'Newest first':
        sort_func = lambda x: os.path.getctime(sd_models.get_closet_checkpoint_match(x).filename)
        checkpoints_list.sort(key=sort_func,reverse=True)
    return checkpoints_list


def get_lora_list():
    """Scan for LoRA files and return list with dtype labels"""
    lora_choices = []

    # Common LoRA locations
    possible_dirs = [
        os.path.join(paths.models_path, 'Lora'),
        os.path.join(paths.models_path, 'lora'),
    ]

    for lora_dir in possible_dirs:
        if os.path.exists(lora_dir):
            for root, dirs, files in os.walk(lora_dir):
                for file in files:
                    if file.endswith('.safetensors'):
                        full_path = os.path.join(root, file)
                        # Get dtype
                        try:
                            with safetensors.torch.safe_open(full_path, framework='pt', device='cpu') as f:
                                keys = list(f.keys())
                                if len(keys) > 0:
                                    tensor = f.get_tensor(keys[0])
                                    dtype_str = str(tensor.dtype).split('.')[1]
                                    # Format: ("display [dtype]", "full_path")
                                    display_name = f"{file} [{dtype_str}]"
                                    lora_choices.append((display_name, full_path))
                        except:
                            # If we can't read it, still add it without dtype
                            lora_choices.append((file, full_path))

    # Sort alphabetically by display name
    lora_choices.sort(key=lambda x: x[0].lower())

    return lora_choices


def refresh_models(sort):
    sd_models.list_models()
    checkpoints_list = get_checkpoints_list(sort)

    return gr.update(choices=checkpoints_list),gr.update(choices=checkpoints_list),gr.update(choices=checkpoints_list),gr.update(choices=checkpoints_list)


### CUSTOM SLIDER FUNCS
def save_custom_sliders(name,*sliders):
    new_preset = {name:sliders}
    with open(custom_sliders_presets,'r') as file:
        sliders_presets = json.load(file)

    sliders_presets .update(new_preset)

    with open(custom_sliders_presets,'w') as file:
            json.dump(sliders_presets,file,indent=0)
    gr.Info('Preset saved')


def get_slider_presets():
    global loaded_slider_presets
    try:
        with open(custom_sliders_presets,'r') as file:
            loaded_slider_presets = json.load(file)
    except FileNotFoundError:
        shutil.copy(custom_sliders_examples,custom_sliders_presets)
        with open(custom_sliders_presets,'r') as file:
            loaded_slider_presets = json.load(file)

    return sorted(list(loaded_slider_presets.keys()))


def load_slider_preset(name):
    preset = loaded_slider_presets[name]
    return [gr.update(value=x) for x in preset]


### LORA MERGE FUNCTIONS ###

def merge_loras_to_checkpoint_ui(checkpoint_name, lora_paths, output_name, strength):
    """Merge multiple LoRAs into a checkpoint sequentially"""
    progress = Progress()

    try:
        if not checkpoint_name:
            return "Error: Please select a base checkpoint"
        if not lora_paths or len(lora_paths) == 0:
            return "Error: Please select at least one LoRA"
        if not output_name:
            return "Error: Please provide an output name"

        # Get checkpoint path
        checkpoint_info = sd_models.get_closet_checkpoint_match(checkpoint_name)
        if not checkpoint_info:
            return f"Error: Could not find checkpoint {checkpoint_name}"

        checkpoint_path = checkpoint_info.filename

        # Determine output path
        output_dir = os.path.dirname(checkpoint_path)
        output_path = os.path.join(output_dir, f"{output_name}.safetensors")

        progress(f"Merging {len(lora_paths)} LoRA(s) to checkpoint...")

        # Load checkpoint once
        with safetensors.torch.safe_open(checkpoint_path, framework='pt', device='cpu') as checkpoint_file:
            checkpoint_keys = list(checkpoint_file.keys())
            checkpoint_dict = {k: checkpoint_file.get_tensor(k) for k in checkpoint_keys}

        progress(f"Loaded base checkpoint: {checkpoint_name}")

        # Apply each LoRA sequentially
        for idx, lora_path in enumerate(lora_paths):
            progress(f"Applying LoRA {idx+1}/{len(lora_paths)}: {os.path.basename(lora_path)}")

            # Load LoRA
            with safetensors.torch.safe_open(lora_path, framework='pt', device='cpu') as lora_file:
                lora_keys = list(lora_file.keys())
                lora_dict = {k: lora_file.get_tensor(k) for k in lora_keys}

            # Group LoRA keys
            lora_groups = {}
            for lora_key in lora_keys:
                base_key, lora_type = lora_merge.parse_lora_key(lora_key)
                if base_key is None:
                    continue

                if base_key not in lora_groups:
                    lora_groups[base_key] = {}
                lora_groups[base_key][lora_type] = lora_key

            merged_count = 0
            # Apply LoRA to checkpoint dict
            for base_key, lora_parts in lora_groups.items():
                if 'up' not in lora_parts or 'down' not in lora_parts:
                    continue

                # Find matching checkpoint key
                checkpoint_key = None
                for ck_key in checkpoint_keys:
                    if base_key in ck_key or ck_key in base_key:
                        checkpoint_key = ck_key
                        break

                if checkpoint_key is None or checkpoint_key not in checkpoint_dict:
                    continue

                try:
                    lora_up = lora_dict[lora_parts['up']]
                    lora_down = lora_dict[lora_parts['down']]

                    if 'alpha' in lora_parts:
                        alpha = lora_dict[lora_parts['alpha']].item()
                    else:
                        alpha = float(lora_down.shape[0])

                    rank = lora_down.shape[0]
                    lora_delta = (lora_up @ lora_down) * (alpha / rank) * strength

                    original_weight = checkpoint_dict[checkpoint_key]
                    if original_weight.shape == lora_delta.shape:
                        checkpoint_dict[checkpoint_key] = original_weight + lora_delta
                        merged_count += 1

                except Exception:
                    continue

            progress(f"  Merged {merged_count} layers from LoRA {idx+1}")

        progress(f"Saving merged checkpoint to: {output_path}")
        safetensors.torch.save_file(checkpoint_dict, output_path)

        # Refresh model list
        sd_models.list_models()

        progress(f"✓ Successfully merged {len(lora_paths)} LoRA(s) into checkpoint!", popup=True)
        return progress.get_report()

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}"


def merge_loras_ui(lora1, weight1, lora2, weight2, lora3, weight3, output_name):
    progress = Progress()

    try:
        # Collect LoRA paths and weights
        lora_paths = []
        weights = []

        if lora1 and os.path.exists(lora1):
            lora_paths.append(lora1)
            weights.append(weight1)
        if lora2 and os.path.exists(lora2):
            lora_paths.append(lora2)
            weights.append(weight2)
        if lora3 and os.path.exists(lora3):
            lora_paths.append(lora3)
            weights.append(weight3)

        if len(lora_paths) < 2:
            return "Error: Please provide at least 2 valid LoRA paths"

        if not output_name:
            return "Error: Please provide an output name"

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight == 0:
            return "Error: Total weight cannot be zero"
        weights = [w / total_weight for w in weights]

        # Determine output path (use same dir as first LoRA)
        output_dir = os.path.dirname(lora_paths[0])
        output_path = os.path.join(output_dir, f"{output_name}.safetensors")

        # Merge
        result = lora_merge.merge_loras(
            lora_paths,
            output_path,
            weights=weights,
            progress=progress
        )

        return progress.get_report()

    except Exception as e:
        return f"Error: {str(e)}"
