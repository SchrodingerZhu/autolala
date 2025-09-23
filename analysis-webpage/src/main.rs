use std::time::Duration;

use charming::WasmRenderer;
use denning::MissRatioCurve;
use serde::{Deserialize, Serialize};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::spawn_local;
use web_sys::{
    HtmlInputElement, HtmlSelectElement, HtmlSpanElement,
    js_sys::{self},
    window,
};
use yew::prelude::*;

#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOCATOR: talc::TalckWasm = unsafe { talc::TalckWasm::new_global() };

#[derive(Serialize)]
pub enum AnalysisRequest {
    Barvinok {
        source: String,
        approximation: bool,
        infinite_repeat: bool,
        block_size: Option<usize>,
        lower_bounds: Vec<usize>,
    },
    Salt {
        source: String,
        block_size: Option<usize>,
    },
}

#[derive(Deserialize)]
struct BarvinokResult {
    ri_values: Box<[String]>,
    symbol_ranges: Box<[String]>,
    counts: Box<[String]>,
    portions: Box<[String]>,
    total_count: String,
    miss_ratio_curve: MissRatioCurve,
    analysis_time: Duration,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct SaltResult {
    ri_values: Vec<String>,
    portions: Vec<String>,
    total_count: String,
    miss_ratio_curve: MissRatioCurve,
    analysis_time: Duration,
}

pub fn get_ace_editor_text() -> Option<String> {
    let window = window()?;
    let ace = js_sys::Reflect::get(&window, &JsValue::from_str("ace")).ok()?;

    // Call `ace.edit("editor")`
    let editor_val = js_sys::Reflect::get(&ace, &JsValue::from_str("edit")).ok()?;
    let editor_fn = editor_val.dyn_ref::<js_sys::Function>()?;
    let editor = editor_fn
        .call1(&JsValue::NULL, &JsValue::from_str("editor"))
        .ok()?;

    // Call `editor.getValue()`
    let get_value_fn = js_sys::Reflect::get(&editor, &JsValue::from_str("getValue")).ok()?;
    let get_value_fn = get_value_fn.dyn_ref::<js_sys::Function>()?;
    let value = get_value_fn.call0(&editor).ok()?;

    value.as_string()
}

pub fn get_selected_solver() -> Option<String> {
    let document = window()?.document()?;
    let select_element = document
        .get_element_by_id("solver")?
        .dyn_into::<HtmlSelectElement>()
        .ok()?;
    Some(select_element.value())
}

pub fn get_barvinok_block_size() -> Option<usize> {
    let document = window()?.document()?;
    let input = document
        .get_element_by_id("barvinok-block-size")?
        .dyn_into::<HtmlInputElement>()
        .ok()?;
    input.value().parse().ok()
}

pub fn get_salt_block_size() -> Option<usize> {
    let document = window()?.document()?;
    let input = document
        .get_element_by_id("salt-block-size")?
        .dyn_into::<HtmlInputElement>()
        .ok()?;
    input.value().parse().ok()
}

pub fn get_barvinok_lower_bounds() -> Option<Vec<usize>> {
    let document = window()?.document()?;
    let input = document
        .get_element_by_id("barvinok-lower-bounds")?
        .dyn_into::<HtmlInputElement>()
        .ok()?;
    input
        .value()
        .split(',')
        .map(|s| s.trim().parse::<usize>().ok())
        .collect()
}

pub fn get_barvinok_approximation() -> Option<bool> {
    let document = window()?.document()?;
    let input = document
        .get_element_by_id("barvinok-approximation")?
        .dyn_into::<HtmlInputElement>()
        .ok()?;
    Some(input.checked())
}

pub fn get_barvinok_infinite_repeat() -> Option<bool> {
    let document = window()?.document()?;
    let input = document
        .get_element_by_id("barvinok-infinite-repeat")?
        .dyn_into::<HtmlInputElement>()
        .ok()?;
    Some(input.checked())
}

pub fn set_total_count(value: &str) {
    if let Some(document) = window().and_then(|w| w.document())
        && let Some(span) = document
            .get_element_by_id("total-count")
            .and_then(|el| el.dyn_into::<HtmlSpanElement>().ok())
    {
        span.set_inner_html(value);
    }
}

pub fn set_analysis_time(value: &str) {
    if let Some(document) = window().and_then(|w| w.document())
        && let Some(span) = document
            .get_element_by_id("analysis-time")
            .and_then(|el| el.dyn_into::<HtmlSpanElement>().ok())
    {
        span.set_inner_html(value);
    }
}

pub fn clear_table() {
    if let Some(document) = window().and_then(|w| w.document())
        && let Some(container) = document.get_element_by_id("ri-table")
    {
        container.set_inner_html("");
    }
}

pub fn clear_canvas() {
    let canvas = window()
        .unwrap()
        .document()
        .unwrap()
        .get_element_by_id("mrc-container")
        .unwrap();
    canvas.set_inner_html(r#"<div id="miss-ratio-curve"></div>"#);
}

fn string_to_mathjax_inner(s: &str, buffer: &mut String) {
    let mut prev_is_less = false;
    let mut prev_is_greater = false;
    let mut last_is_alpha = false;
    let mut need_to_close = false;
    for c in s.chars() {
        if !c.is_numeric() && need_to_close {
            buffer.push('}');
            need_to_close = false;
        }
        match c {
            '*' => buffer.push_str("\\times "),
            c if c.is_numeric() && last_is_alpha => {
                buffer.push_str("_{");
                buffer.push(c);
                need_to_close = true;
                last_is_alpha = false;
            }
            c if c.is_alphabetic() => {
                buffer.push(c);
                last_is_alpha = true;
            }
            '=' => {
                if prev_is_less {
                    buffer.pop();
                    buffer.push_str("\\le ");
                    prev_is_less = false;
                } else if prev_is_greater {
                    buffer.pop();
                    buffer.push_str("\\ge ");
                    prev_is_greater = false;
                } else {
                    buffer.push('=');
                }
                last_is_alpha = false;
            }
            '<' => {
                buffer.push('<');
                prev_is_less = true;
                last_is_alpha = false;
            }
            '>' => {
                buffer.push('>');
                prev_is_greater = true;
                last_is_alpha = false;
            }
            _ => {
                buffer.push(c);
                last_is_alpha = false;
                prev_is_greater = false;
                prev_is_less = false;
            }
        }
    }
    if need_to_close {
        buffer.push('}');
    }
}

pub fn render_canvas(mrc: &MissRatioCurve) {
    let mrc_container = window()
        .unwrap()
        .document()
        .unwrap()
        .get_element_by_id("miss-ratio-curve")
        .unwrap();
    let width = mrc_container.client_width() as u32;
    let height = (0.618 * width as f64).ceil() as u32;
    let render = WasmRenderer::new(width, height);
    let chart = mrc.plot_interactive_miss_ratio_curve();
    render.render("miss-ratio-curve", &chart).unwrap();
}

fn ident_to_mathjax(s: &str) -> String {
    let re = regex::Regex::new(r"([a-zA-Z])(\d+)").unwrap();
    re.replace_all(s, "${1}_{${2}}").to_string()
}

fn string_to_mathjax(s: &str) -> String {
    if s.trim().is_empty() {
        return String::new();
    }
    let mut buffer = String::new();
    string_to_mathjax_inner(s, &mut buffer);
    // replace and with \wedge and or with \vee
    buffer = buffer.replace(" and ", " \\wedge ");
    buffer = buffer.replace(" or ", " \\vee ");
    // replace mod with \\bmod
    buffer = buffer.replace(" mod ", " \\bmod ");
    buffer
}

pub fn render_to_string(s: &str) -> String {
    let render = js_sys::Reflect::get(
        &js_sys::Reflect::get(&window().unwrap(), &JsValue::from_str("katex")).unwrap(),
        &JsValue::from_str("renderToString"),
    )
    .unwrap()
    .dyn_into::<js_sys::Function>()
    .unwrap();
    let input = JsValue::from_str(s);

    render
        .call1(&JsValue::NULL, &input)
        .unwrap()
        .as_string()
        .unwrap()
}

pub fn update_ri_table_barvinok(
    ri_values: &[String],
    symbol_ranges: &[String],
    counts: &[String],
    portions: &[String],
) -> Result<(), JsValue> {
    // Ensure all vectors have the same length
    let len = ri_values.len();
    if symbol_ranges.len() != len || counts.len() != len || portions.len() != len {
        return Err(JsValue::from_str("Input arrays must have the same length"));
    }

    // Start building HTML
    let mut html = String::new();
    html.push_str("<table class=\"table table-bordered table-striped table-sm\">");
    html.push_str("<thead><tr><th>Reuse Interval</th><th>Symbol Range</th><th>Count</th><th>Portion</th></tr></thead><tbody>");

    for i in 0..len {
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            render_to_string(&ident_to_mathjax(&ri_values[i])),
            render_to_string(&string_to_mathjax(&symbol_ranges[i])),
            render_to_string(&ident_to_mathjax(&counts[i])),
            render_to_string(&ident_to_mathjax(&portions[i]))
        ));
    }

    html.push_str("</tbody></table>");

    // Set it as the innerHTML of the target element
    let document = window().unwrap().document().unwrap();
    let container = document
        .get_element_by_id("ri-table")
        .ok_or_else(|| JsValue::from_str("Element with id 'ri-table' not found"))?;

    container.set_inner_html(&html);

    Ok(())
}

pub fn update_ri_table_salt(ri_values: &[String], portions: &[String]) -> Result<(), JsValue> {
    // Ensure all vectors have the same length
    let len = ri_values.len();
    if portions.len() != len {
        return Err(JsValue::from_str("Input arrays must have the same length"));
    }

    // Start building HTML
    let mut html = String::new();
    html.push_str("<table class=\"table table-bordered table-striped table-sm\">");
    html.push_str("<thead><tr><th>Reuse Interval</th><th>Portion</th></tr></thead><tbody>");

    for i in 0..len {
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td></tr>",
            render_to_string(&ident_to_mathjax(&ri_values[i])),
            render_to_string(&ident_to_mathjax(&portions[i]))
        ));
    }

    html.push_str("</tbody></table>");

    // Set it as the innerHTML of the target element
    let document = window().unwrap().document().unwrap();
    let container = document
        .get_element_by_id("ri-table")
        .ok_or_else(|| JsValue::from_str("Element with id 'ri-table' not found"))?;

    container.set_inner_html(&html);

    Ok(())
}

#[function_component]
fn App() -> Html {
    let show_barvinok = use_state(|| true);
    let show_error = use_state(|| false);
    let error_message = use_state(String::new);

    let onchange = {
        let show_barvinok = show_barvinok.clone();
        Callback::from(move |event: Event| {
            let select_elem = event.target_unchecked_into::<HtmlSelectElement>();
            let selected_value = select_elem.value();
            show_barvinok.set(selected_value == "Barvinok");
        })
    };

    let onclick = {
        let show_error = show_error.clone();
        let error_message = error_message.clone();
        move |_| {
            clear_table();
            set_total_count("Loading...");
            set_analysis_time("Loading...");
            clear_canvas();
            show_error.set(false);
            let text = get_ace_editor_text()
                .and_then(|s| if s.trim().is_empty() { None } else { Some(s) });
            if text.is_none() {
                error_message.set("Please enter MLIR code".to_string());
                show_error.set(true);
                return;
            }
            let method = get_selected_solver();
            match method.as_deref() {
                Some("Barvinok") => {
                    let block_size = get_barvinok_block_size().unwrap_or(1);
                    let lower_bounds = get_barvinok_lower_bounds().unwrap_or_default();
                    let approximation = get_barvinok_approximation().unwrap_or_default();
                    let infinite_repeat = get_barvinok_infinite_repeat().unwrap_or_default();
                    let request = AnalysisRequest::Barvinok {
                        source: text.unwrap(),
                        approximation,
                        infinite_repeat,
                        block_size: Some(block_size),
                        lower_bounds,
                    };
                    let error_message = error_message.clone();
                    let show_error = show_error.clone();
                    spawn_local(async move {
                        let future = async {
                            let response = gloo_net::http::Request::post("/analysis")
                                .header("Content-Type", "application/json")
                                .json(&request)?;
                            let response = response.send().await?;
                            if response.status() == 200 {
                                let result = response.json::<BarvinokResult>().await?;
                                set_analysis_time(&format!(
                                    "{} ms",
                                    result.analysis_time.as_millis()
                                ));
                                set_total_count(&render_to_string(&string_to_mathjax(
                                    &result.total_count,
                                )));
                                render_canvas(&result.miss_ratio_curve);
                                update_ri_table_barvinok(
                                    &result.ri_values,
                                    &result.symbol_ranges,
                                    &result.counts,
                                    &result.portions,
                                )
                                .unwrap();
                            } else {
                                let body = response.text().await.unwrap_or_else(|e| format!("{e}"));
                                let body = strip_ansi_escapes::strip_str(&body);
                                error_message.set(body);
                                show_error.set(true);
                            }
                            anyhow::Ok(())
                        };
                        future.await.unwrap_or_else(|err| {
                            error_message.set(format!("Error: {err}"));
                            show_error.set(true);
                        });
                    })
                }
                Some("Salt") => {
                    let block_size = get_salt_block_size();
                    let request = AnalysisRequest::Salt {
                        source: text.unwrap(),
                        block_size,
                    };
                    let error_message = error_message.clone();
                    let show_error = show_error.clone();
                    spawn_local(async move {
                        let future = async {
                            let response = gloo_net::http::Request::post("/analysis")
                                .header("Content-Type", "application/json")
                                .json(&request)?;
                            let response = response.send().await?;
                            if response.status() == 200 {
                                let result = response.json::<SaltResult>().await?;
                                set_total_count(&render_to_string(&string_to_mathjax(
                                    &result.total_count,
                                )));
                                set_analysis_time(&format!(
                                    "{} ms",
                                    result.analysis_time.as_millis()
                                ));
                                update_ri_table_salt(&result.ri_values, &result.portions).unwrap();
                                render_canvas(&result.miss_ratio_curve);
                            } else {
                                let body = response.text().await.unwrap_or_else(|e| format!("{e}"));
                                let body = strip_ansi_escapes::strip_str(&body);
                                error_message.set(body);
                                show_error.set(true);
                            }
                            anyhow::Ok(())
                        };
                        future.await.unwrap_or_else(|err| {
                            error_message.set(format!("Error: {err}"));
                            show_error.set(true);
                        });
                    })
                }
                _ => {
                    error_message.set("invalid solver".to_string());
                    show_error.set(true);
                }
            }
        }
    };

    let example_code = include_str!("../../analyzer/misc/const_matmul_4acc.mlir");
    use_effect(|| {
        // Run this only once after the component is mounted
        if let Some(window) = window()
            && let Ok(ace_ns) = js_sys::Reflect::get(&window, &JsValue::from_str("ace"))
        {
            let editor = js_sys::Reflect::get(&ace_ns, &JsValue::from_str("edit"))
                .ok()
                .and_then(|edit_fn| {
                    edit_fn
                        .dyn_ref::<js_sys::Function>()
                        .and_then(|f| f.call1(&JsValue::NULL, &JsValue::from_str("editor")).ok())
                });

            if let Some(editor_obj) = editor {
                // editor.setTheme("ace/theme/monokai")
                let _ = js_sys::Reflect::get(&editor_obj, &JsValue::from_str("setTheme"))
                    .ok()
                    .and_then(|set_theme_fn| {
                        set_theme_fn.dyn_ref::<js_sys::Function>().and_then(|f| {
                            f.call1(&editor_obj, &JsValue::from_str("ace/theme/monokai"))
                                .ok()
                        })
                    });

                // editor.session.setMode("ace/mode/javascript")
                let _ = js_sys::Reflect::get(&editor_obj, &JsValue::from_str("session"))
                    .ok()
                    .and_then(|session| {
                        js_sys::Reflect::get(&session, &JsValue::from_str("setMode"))
                            .ok()
                            .and_then(|set_mode_fn| {
                                set_mode_fn.dyn_ref::<js_sys::Function>().and_then(|f| {
                                    f.call1(&session, &JsValue::from_str("ace/mode/plain_text"))
                                        .ok()
                                })
                            })
                    });
            }
        }

        || ()
    });

    html! {
    <>
        <div class="container">
            // title
            <h1 class="text-center">{"Affine Loop Locality Analysis"}</h1>
            <div class="gap-3 mb-3">
                <div id="error-message" class="alert alert-danger" role="alert" style={if *show_error { "white-space: pre-wrap;" } else { "display: none;" }}>
                    <strong id="error-message-text">{(*error_message).clone()}</strong>
                </div>
                <div id="error-message" class="alert alert-info" role="alert">
                    <strong id="error-message-text">{"Symbol subscripts are in the order of introduction. Salt solver can only handle perfectly nested loop but it accept symbols appearing as coefficients. Barvinok solver can only handle pure affine expressions. If there is no symbolic input, a miss ratio curve will be produced."}</strong>
                </div>
                <div class="input-group">
                    <span class="input-group-text"> { "Solver" }</span>
                    <select id="solver" class="form-select form-select-lg" onchange={onchange}>
                        <option value="Barvinok" selected=true>{"Barvinok"}</option>
                        <option value="Salt">{"Salt"}</option>
                    </select>
                </div>
                <br />
                <div id="barvinok-options" style={if *show_barvinok { "" } else { "display: none;" }}>
                    <div class="input-group">
                        <span class="input-group-text"> { "Block Size" }</span>
                        <input type="text" class="form-control" id="barvinok-block-size" placeholder="1" required=true />
                    </div>
                    <br />
                    <div class="input-group">
                        <span class="input-group-text"> { "Lower Bounds" }</span>
                        <input type="text" class="form-control" id="barvinok-lower-bounds" placeholder=""/>
                    </div>
                    <br />
                    <div class="form-check form-switch">
                        <input class="form-check-input" type="checkbox" value="" id="barvinok-approximation"/>
                        <label class="form-check-label" for="barvinok-approximation">
                            {"Approximation"}
                        </label>
                    </div>
                    <div class="form-check form-switch">
                        <input class="form-check-input" type="checkbox" value="" id="barvinok-infinite-repeat"/>
                        <label class="form-check-label" for="barvinok-infinite-repeat">
                            {"Infinite Repeat"}
                        </label>
                    </div>
                </div>
                <div id="salt-options" style={if *show_barvinok  { "display: none;" } else { "" }}>
                    <div class="input-group">
                        <span class="input-group-text"> { "Block Size" }</span>
                        <input type="text" class="form-control" id="salt-block-size" placeholder="" required=false />
                    </div>
                </div>
                <br />
                <button type="button" class="btn btn-primary" {onclick}>{ "Analyze" }</button>
                <hr />
                <p type="text"> { "MLIR Source Code" }</p>
                <div id="editor"> {example_code} </div>
                <hr />
                <div class="input-group">
                    <span class="input-group-text"> { "Total Count" }</span>
                    <span class="form-control" id="total-count"></span>
                </div>
                <br />
                <div class="input-group">
                    <span class="input-group-text"> { "Analysis Time" }</span>
                    <span class="form-control" id="analysis-time"></span>
                </div>
                <br />
                // RI table div
                <div id="ri-table"></div>
                <div id="mrc-container" class="container text-center">
                    <div id="miss-ratio-curve"></div>
                </div>
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.8/dist/umd/popper.min.js" integrity="sha384-I7E8VVD/ismYTF4hNIPjVp/Zjvgyol6VFvRkX/vR+Vc4jQkC+hVqc2pM8ODewa9r" crossorigin="anonymous">
        </script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.6/dist/js/bootstrap.min.js" integrity="sha384-RuyvpeZCxMJCqVUGFI0Do1mQrods/hhxYlcVfGPOfQtPJh0JCw12tUAZ/Mv10S7D" crossorigin="anonymous"></script>
        <script>
        </script>
    </>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
