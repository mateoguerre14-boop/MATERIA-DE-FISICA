import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint

st.set_page_config(page_title="Laboratorio MAS", layout="wide", page_icon="⚛️")

def plot_time_series(t, y_data, names, title, y_label):
    fig = go.Figure()
    for y, name in zip(y_data, names):
        fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name=name))
    fig.update_layout(
        title=title, xaxis_title="Tiempo (s)", yaxis_title=y_label,
        hovermode="x unified", height=400, template="plotly_white"
    )
    return fig

st.sidebar.title("Menú de Simulación")
opcion = st.sidebar.radio(
    "Selecciona un módulo:",
    ("Inicio", "Masa-Resorte", "Péndulo Simple", "Análisis T vs (m, k)", "MAS Amortiguado")
)

if opcion == "Inicio":
    st.title("Movimiento Armónico Simple (MAS)")
    st.markdown("""
    Bienvenido a la aplicación interactiva.
    
    ### Instrucciones para validación:
    1. Selecciona un módulo en la izquierda.
    2. Ingresa los parámetros que usaste en tus cálculos a mano.
    3. Compara los **"Resultados Calculados"** que aparecen en pantalla con tu hoja.
    """)
    st.info("Selecciona una opción en el menú de la izquierda para comenzar.")

elif opcion == "Masa-Resorte":
    st.header("Sistema Masa-Resorte")
    
    tipo_sistema = st.radio("Orientación:", ("Horizontal", "Vertical"), horizontal=True)
    st.divider()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Entrada de Datos")
        m = st.number_input("Masa (m) [kg]", value=1.0, step=0.1)
        k = st.number_input("Constante (k) [N/m]", value=16.0, step=1.0)
        A = st.number_input("Amplitud (A) [m]", value=1.0, step=0.1)
        duration = st.slider("Tiempo simulación [s]", 5, 30, 10)
    
    w = np.sqrt(k / m)
    T = 2 * np.pi / w
    t = np.linspace(0, duration, 500)
    
    x = A * np.cos(w * t)
    v = -A * w * np.sin(w * t)
    a = -A * (w**2) * np.cos(w * t)
    E_total = 0.5 * k * A**2
    
    E_p = 0.5 * k * x**2
    E_k = 0.5 * m * v**2

    with col2:
        st.subheader("Resultados para Validar")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Frecuencia (ω)", f"{w:.2f} rad/s", help="Calculado: √(k/m)")
        res_col2.metric("Periodo (T)", f"{T:.2f} s", help="Calculado: 2π/ω")
        res_col3.metric("Energía Total (E)", f"{E_total:.2f} J", help="Calculado: 0.5 * k * A²")
        
        st.divider()

        if tipo_sistema == "Vertical":
            g = 9.81
            delta_eq = (m * g) / k
            st.warning(f"Vertical: El resorte se estira {delta_eq:.3f} m por el peso.")
        else:
            st.success("Horizontal: Movimiento oscilatorio ideal.")

        # Gráficas
        tab1, tab2 = st.tabs(["Gráfica Cinemática", "Gráfica de Energía"])
        with tab1:
            st.plotly_chart(plot_time_series(t, [x, v], ["Posición (m)", "Velocidad (m/s)"], "Cinemática", "Valor"), use_container_width=True)
        with tab2:
            st.plotly_chart(plot_time_series(t, [E_p, E_k, E_p+E_k], ["Potencial", "Cinética", "Mecánica (Total)"], "Energía", "Joules"), use_container_width=True)

elif opcion == "Péndulo Simple":
    st.header("Péndulo Simple")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parámetros")
        L = st.number_input("Longitud (L) [m]", value=1.0, step=0.1)
        g = st.number_input("Gravedad (g) [m/s²]", value=9.81)
        theta0_deg = st.slider("Ángulo inicial (°)", 1, 179, 5)
        theta0 = np.radians(theta0_deg)
        
    w0 = np.sqrt(g / L)
    T_teorico = 2 * np.pi * np.sqrt(L / g)
    
    t = np.linspace(0, 10, 500)
    theta_linear = theta0 * np.cos(w0 * t)
    
    def pendulum_ode(y, t, g, L):
        theta, omega = y
        dydt = [omega, -(g/L) * np.sin(theta)]
        return dydt

    sol = odeint(pendulum_ode, [theta0, 0.0], t, args=(g, L))
    theta_nonlinear = sol[:, 0]
    
    with col2:
        st.subheader("📋 Resultados Teóricos (Ángulos Pequeños)")
        c1, c2 = st.columns(2)
        c1.metric("Frecuencia Natural (ω)", f"{w0:.3f} rad/s")
        c2.metric("Periodo Teórico (T)", f"{T_teorico:.3f} s")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=np.degrees(theta_linear), name='Modelo Lineal (Fórmula)', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=t, y=np.degrees(theta_nonlinear), name='Modelo Real (No Lineal)'))
        fig.update_layout(title="Comparación de Modelos", xaxis_title="Tiempo (s)", yaxis_title="Ángulo (grados)")
        st.plotly_chart(fig, use_container_width=True)

elif opcion == "Análisis T vs (m, k)":
    st.header("Análisis de Periodo")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### T vs Masa (m)")
        k_val = st.slider("Valor fijo de k", 1.0, 50.0, 10.0)
        ms = np.linspace(0.1, 10, 100)
        Ts = 2*np.pi*np.sqrt(ms/k_val)
        fig = go.Figure(go.Scatter(x=ms, y=Ts))
        fig.update_layout(xaxis_title="Masa (kg)", yaxis_title="Periodo (s)", title=f"Efecto de Masa (k={k_val})")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("### T vs Constante (k)")
        m_val = st.slider("Valor fijo de m", 0.1, 10.0, 1.0)
        ks = np.linspace(1.0, 50, 100)
        Ts_k = 2*np.pi*np.sqrt(m_val/ks)
        fig2 = go.Figure(go.Scatter(x=ks, y=Ts_k, line=dict(color='red')))
        fig2.update_layout(xaxis_title="Constante k (N/m)", yaxis_title="Periodo (s)", title=f"Efecto de k (m={m_val})")
        st.plotly_chart(fig2, use_container_width=True)

elif opcion == "MAS Amortiguado":
    st.header("Oscilador Amortiguado")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parámetros")
        m = st.number_input("Masa (m)", value=1.0)
        k = st.number_input("Constante (k)", value=10.0)
        b = st.slider("Amortiguamiento (b)", 0.0, 5.0, 1.0, 0.1)
        A_inicial = 1.0
        
        st.divider()
        st.markdown("**Verificación Puntual**")
        t_check = st.number_input("Tiempo a verificar (s)", value=2.0)
        
    def damped_ode(y, t, m, k, b):
        x, v = y
        dydt = [v, -(b/m)*v - (k/m)*x]
        return dydt
    
    t = np.linspace(0, 20, 1000)
    sol = odeint(damped_ode, [A_inicial, 0.0], t, args=(m, k, b))
    x_damped = sol[:, 0]
    
    envelope = A_inicial * np.exp(-(b / (2*m)) * t)
    
    val_envelope_check = A_inicial * np.exp(-(b / (2*m)) * t_check)

    with col2:
        st.subheader("📋 Validación de Resultados")
                
        met1, met2 = st.columns(2)
        met1.metric(f"Amplitud Teórica en t={t_check}", f"{val_envelope_check:.4f} m", help="Calculado con fórmula exponencial")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=x_damped, name='Posición Real', line=dict(width=2)))
        if b > 0:
            fig.add_trace(go.Scatter(x=t, y=envelope, name='Envolvente (+)', line=dict(dash='dot', color='gray')))
            fig.add_trace(go.Scatter(x=[t_check], y=[val_envelope_check], mode='markers', marker=dict(color='red', size=10), name=f'Punto Verificación t={t_check}'))
            
        fig.update_layout(title="Decaimiento de la Oscilación", xaxis_title="Tiempo (s)", yaxis_title="Posición (m)")
        st.plotly_chart(fig, use_container_width=True)
