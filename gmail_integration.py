"""
PostSale — Integración Gmail
==============================
Módulo que conecta Gmail de un usuario con PostSale para analizar
emails de clientes automáticamente.

Flujo:
  1. El gestor autoriza PostSale con su Gmail (OAuth2)
  2. PostSale lee los emails de clientes en su bandeja
  3. Analiza cada conversación con la IA
  4. Si detecta riesgo Alto o Crítico, crea tarea y alerta

Integración en api.py:
  - Agregar los 3 endpoints de Gmail al final de api.py
  - Agregar las variables de entorno GOOGLE_CLIENT_ID y GOOGLE_CLIENT_SECRET

Uso local:
  1. $env:GOOGLE_CLIENT_ID="tu-client-id"
  2. $env:GOOGLE_CLIENT_SECRET="tu-client-secret"
  3. uvicorn api:app --reload
  4. Ir a http://localhost:8000/gmail/autorizar

Autor: PostSale Engineering
Versión: 0.8.0 (Integración Gmail)
"""

import base64
import email
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

log = logging.getLogger("postsale.gmail")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
]

GMAIL_TOKEN_FILE = "gmail_token.json"

GMAIL_CONFIG = {
    "max_emails_por_cliente": 10,    # últimos N emails por remitente
    "dias_hacia_atras": 30,          # analizar emails de los últimos X días
    "min_emails_para_analizar": 2,   # mínimo de emails para considerar un cliente
}


# ---------------------------------------------------------------------------
# AUTENTICACIÓN OAuth2
# ---------------------------------------------------------------------------

def get_google_flow() -> Flow:
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError("Falta GOOGLE_CLIENT_ID o GOOGLE_CLIENT_SECRET.")

    client_config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost:8000/gmail/callback"],
        }
    }

    flow = Flow.from_client_config(
        client_config,
        scopes=GMAIL_SCOPES,
        redirect_uri="http://localhost:8000/gmail/callback",
    )
    flow.code_verifier = None
    return flow

def get_gmail_service():
    """
    Retorna el servicio de Gmail autenticado.
    Si el token existe y es válido, lo reutiliza.
    Si expiró, lo renueva automáticamente.
    """
    creds = None

    if os.path.exists(GMAIL_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, GMAIL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Guardar el token renovado
            with open(GMAIL_TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
        else:
            raise ValueError(
                "Gmail no está autorizado. "
                "Ir a /gmail/autorizar para conectar tu cuenta."
            )

    return build("gmail", "v1", credentials=creds)


def gmail_esta_autorizado() -> bool:
    """Verifica si Gmail está autorizado y el token es válido."""
    if not os.path.exists(GMAIL_TOKEN_FILE):
        return False
    try:
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, GMAIL_SCOPES)
        return creds and (creds.valid or (creds.expired and creds.refresh_token))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# LECTURA DE EMAILS
# ---------------------------------------------------------------------------

def extraer_texto_email(mensaje: dict) -> str:
    """Extrae el texto plano de un mensaje de Gmail."""
    try:
        payload = mensaje.get("payload", {})
        parts = payload.get("parts", [])

        # Email simple sin partes
        if not parts:
            body = payload.get("body", {})
            data = body.get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
            return ""

        # Email con múltiples partes — buscar text/plain
        for part in parts:
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

        # Si no hay text/plain, buscar recursivamente
        for part in parts:
            sub_parts = part.get("parts", [])
            for sub_part in sub_parts:
                if sub_part.get("mimeType") == "text/plain":
                    data = sub_part.get("body", {}).get("data", "")
                    if data:
                        return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

    except Exception as e:
        log.warning(f"Error extrayendo texto del email: {e}")

    return ""


def extraer_remitente(mensaje: dict) -> tuple[str, str]:
    """
    Extrae el nombre y email del remitente.
    Retorna (nombre, email).
    """
    headers = mensaje.get("payload", {}).get("headers", [])
    from_header = next((h["value"] for h in headers if h["name"] == "From"), "")

    if "<" in from_header:
        partes = from_header.split("<")
        nombre = partes[0].strip().strip('"')
        email_addr = partes[1].strip().rstrip(">")
    else:
        nombre = from_header
        email_addr = from_header

    return nombre, email_addr


def extraer_asunto(mensaje: dict) -> str:
    """Extrae el asunto del email."""
    headers = mensaje.get("payload", {}).get("headers", [])
    return next((h["value"] for h in headers if h["name"] == "Subject"), "Sin asunto")


def obtener_emails_clientes(
    service,
    dias_hacia_atras: int = None,
) -> dict[str, list[dict]]:
    """
    Lee los emails de la bandeja de entrada y los agrupa por remitente.

    Returns:
        Diccionario {email_remitente: [lista de mensajes]}
    """
    if dias_hacia_atras is None:
        dias_hacia_atras = GMAIL_CONFIG["dias_hacia_atras"]

    fecha_desde = (datetime.now() - timedelta(days=dias_hacia_atras)).strftime("%Y/%m/%d")
    query = f"in:inbox after:{fecha_desde} -from:me"

    try:
        resultado = service.users().messages().list(
            userId="me",
            q=query,
            maxResults=200,
        ).execute()

        mensajes_ids = resultado.get("messages", [])
        log.info(f"Gmail: {len(mensajes_ids)} emails encontrados en los últimos {dias_hacia_atras} días")

        emails_por_remitente = {}

        for msg_ref in mensajes_ids:
            try:
                mensaje = service.users().messages().get(
                    userId="me",
                    id=msg_ref["id"],
                    format="full",
                ).execute()

                nombre, email_addr = extraer_remitente(mensaje)
                asunto = extraer_asunto(mensaje)
                texto = extraer_texto_email(mensaje)

                if not texto or len(texto.strip()) < 10:
                    continue

                if email_addr not in emails_por_remitente:
                    emails_por_remitente[email_addr] = []

                emails_por_remitente[email_addr].append({
                    "nombre_remitente": nombre,
                    "email": email_addr,
                    "asunto": asunto,
                    "texto": texto[:1500],  # limitamos para no sobrecargar el prompt
                    "fecha": mensaje.get("internalDate", ""),
                })

            except Exception as e:
                log.warning(f"Error procesando email {msg_ref['id']}: {e}")
                continue

        log.info(f"Gmail: {len(emails_por_remitente)} remitentes únicos encontrados")
        return emails_por_remitente

    except HttpError as e:
        log.error(f"Error leyendo Gmail: {e}")
        raise


# ---------------------------------------------------------------------------
# ANÁLISIS DE CONVERSACIONES
# ---------------------------------------------------------------------------

def construir_contexto_conversacion(emails: list[dict]) -> str:
    """
    Construye un resumen de la conversación de un cliente
    para enviarlo al motor de IA.
    """
    emails_ordenados = sorted(
        emails,
        key=lambda e: e.get("fecha", "0"),
        reverse=True,
    )[:GMAIL_CONFIG["max_emails_por_cliente"]]

    partes = []
    for i, email_data in enumerate(emails_ordenados, 1):
        partes.append(
            f"Email {i} — Asunto: {email_data['asunto']}\n"
            f"Contenido: {email_data['texto'][:500]}"
        )

    return "\n\n---\n\n".join(partes)


async def analizar_emails_cliente(
    nombre_empresa: str,
    emails: list[dict],
    analizar_fn,
    coleccion=None,
) -> Optional[dict]:
    """
    Analiza los emails de un cliente con el motor de IA.

    Args:
        nombre_empresa: Nombre o email del cliente
        emails: Lista de emails del cliente
        analizar_fn: Función de análisis (analizar_con_rag o analizar_con_ia)
        coleccion: Colección RAG (opcional)

    Returns:
        Resultado del análisis o None si hay error
    """
    from api import InteraccionInput, TipoInteraccion

    if len(emails) < GMAIL_CONFIG["min_emails_para_analizar"]:
        return None

    contexto = construir_contexto_conversacion(emails)

    interaccion = InteraccionInput(
        tipo_interaccion=TipoInteraccion.EMAIL,
        texto_mensaje=f"[Análisis automático Gmail — {len(emails)} emails]\n\n{contexto}",
        dias_desde_ultima_conexion=0,
    )

    try:
        if coleccion:
            resultado = await analizar_fn(
                nombre=nombre_empresa,
                plan="Professional",
                interacciones=[interaccion],
                coleccion=coleccion,
            )
        else:
            resultado = await analizar_fn(
                nombre=nombre_empresa,
                plan="Professional",
                interacciones=[interaccion],
            )
        return resultado
    except Exception as e:
        log.error(f"Error analizando emails de {nombre_empresa}: {e}")
        return None


# ---------------------------------------------------------------------------
# ENDPOINTS FASTAPI — agregar al final de api.py
# ---------------------------------------------------------------------------

ENDPOINTS_GMAIL = '''
# ═══════════════════════════════════════════════════════════════
# GMAIL INTEGRATION — agregar estos imports al inicio de api.py:
#
# from gmail_integration import (
#     get_google_flow, get_gmail_service, gmail_esta_autorizado,
#     obtener_emails_clientes, analizar_emails_cliente,
# )
# ═══════════════════════════════════════════════════════════════


# --- GET /gmail/autorizar — iniciar flujo OAuth2 ---

@app.get("/gmail/autorizar")
def gmail_autorizar():
    """
    Inicia el flujo OAuth2 de Gmail.
    El gestor hace clic en este link y autoriza PostSale.
    """
    from fastapi.responses import RedirectResponse
    try:
        flow = get_google_flow()
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        return RedirectResponse(url=auth_url)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- GET /gmail/callback — recibir el código de Google ---

@app.get("/gmail/callback")
def gmail_callback(code: str, request: Request):
    try:
        import os
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        flow = get_google_flow()
        flow.fetch_token(
            code=code,
            authorization_response=str(request.url),
        )
        creds = flow.credentials

        with open("gmail_token.json", "w") as f:
            f.write(creds.to_json())

        log.info("Gmail autorizado exitosamente")
        return {
            "estado": "ok",
            "mensaje": "Gmail conectado exitosamente. Ya podés usar /gmail/analizar",
            "siguiente_paso": "/gmail/analizar",
        }
    except Exception as e:
        log.error(f"Error en callback de Gmail: {e}")
        raise HTTPException(status_code=400, detail=f"Error autorizando Gmail: {e}")


# --- GET /gmail/estado --- verificar si Gmail está conectado ---

@app.get("/gmail/estado")
def gmail_estado():
    """Verifica si Gmail está autorizado y conectado."""
    autorizado = gmail_esta_autorizado()
    return {
        "autorizado": autorizado,
        "mensaje": "Gmail conectado" if autorizado else "Gmail no conectado — ir a /gmail/autorizar",
        "endpoint_autorizar": "/gmail/autorizar" if not autorizado else None,
    }


# --- POST /gmail/analizar — analizar emails de la bandeja ---

@app.post("/gmail/analizar")
async def gmail_analizar(
    dias: int = 30,
    session: Session = Depends(get_session),
):
    """
    Lee los emails de Gmail de los últimos X días y analiza
    cada conversación con la IA de PostSale.

    Crea clientes automáticamente si no existen en la base de datos.
    Genera tareas y alertas para los que tengan riesgo Alto o Crítico.
    """
    if not gmail_esta_autorizado():
        raise HTTPException(
            status_code=401,
            detail="Gmail no está autorizado. Ir a /gmail/autorizar primero.",
        )

    try:
        service = get_gmail_service()
        emails_por_remitente = obtener_emails_clientes(service, dias_hacia_atras=dias)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo Gmail: {e}")

    resultados = []
    procesados = 0
    omitidos = 0

    for email_addr, emails in emails_por_remitente.items():

        # Buscar si ya existe el cliente en la base de datos
        clientes_existentes = session.exec(
            select(ClienteDB).where(ClienteDB.nombre == email_addr)
        ).all()

        if clientes_existentes:
            cliente = clientes_existentes[0]
        else:
            # Crear cliente nuevo automáticamente
            nombre_display = emails[0].get("nombre_remitente", email_addr) if emails else email_addr
            cliente = ClienteDB(
                nombre=nombre_display or email_addr,
                plan_actual=PlanTipo.PROFESSIONAL,
            )
            session.add(cliente)
            session.commit()
            session.refresh(cliente)
            log.info(f"Cliente creado automáticamente desde Gmail: {cliente.nombre}")

        # Analizar los emails del cliente
        analizar_fn = analizar_con_rag if RAG_DISPONIBLE and coleccion_rag else analizar_con_ia
        coleccion = coleccion_rag if RAG_DISPONIBLE else None

        resultado = await analizar_emails_cliente(
            nombre_empresa=cliente.nombre,
            emails=emails,
            analizar_fn=analizar_fn,
            coleccion=coleccion,
        )

        if resultado is None:
            omitidos += 1
            continue

        # Guardar análisis
        analisis_db = AnalisisDB(
            cliente_id=cliente.id,
            nivel_riesgo=resultado["nivel_riesgo"],
            probabilidad_churn_porcentaje=resultado["probabilidad_churn_porcentaje"],
            razon_principal=resultado["razon_principal"],
            accion_recomendada=resultado["accion_recomendada_para_el_gestor"],
            score_confianza=resultado["score_confianza"],
            intentos_realizados=resultado["intentos_realizados"],
            tiempo_procesamiento_segundos=resultado["tiempo_procesamiento_segundos"],
            requiere_revision_manual=resultado["requiere_revision_manual"],
            interacciones_json=json.dumps(
                [{"tipo": "gmail", "emails": len(emails)}],
                ensure_ascii=False,
            ),
        )
        session.add(analisis_db)
        session.commit()
        session.refresh(analisis_db)

        # Crear tarea si es Alto o Crítico
        if resultado["nivel_riesgo"] in {"Alto", "Crítico"}:
            tarea = TareaDB(
                cliente_id=cliente.id,
                analisis_id=analisis_db.id,
                nivel_riesgo=resultado["nivel_riesgo"],
                accion_sugerida=resultado["accion_recomendada_para_el_gestor"],
                estado="pendiente",
            )
            session.add(tarea)
            session.commit()

            # Enviar alerta email
            resend_key = os.getenv("RESEND_API_KEY")
            if resend_key:
                import resend as resend_lib
                resend_lib.api_key = resend_key
                emoji = "🔴" if resultado["nivel_riesgo"] == "Crítico" else "🟠"
                try:
                    resend_lib.Emails.send({
                        "from": "PostSale <onboarding@resend.dev>",
                        "to": ["santivinas1@gmail.com"],
                        "subject": f"{emoji} PostSale Gmail — {resultado['nivel_riesgo']}: {cliente.nombre}",
                        "html": f"""
                        <div style='font-family:Arial,sans-serif;max-width:560px;margin:0 auto'>
                          <h2>{emoji} Alerta {resultado['nivel_riesgo']} detectada en Gmail</h2>
                          <p><strong>Cliente:</strong> {cliente.nombre}</p>
                          <p><strong>Emails analizados:</strong> {len(emails)}</p>
                          <p><strong>Probabilidad de cancelación:</strong> {resultado['probabilidad_churn_porcentaje']}%</p>
                          <p><strong>Causa:</strong> {resultado['razon_principal']}</p>
                          <div style='background:#fef2f2;border-left:3px solid #dc2626;padding:14px;border-radius:4px;margin-top:16px'>
                            <strong>Acción recomendada:</strong><br><br>
                            {resultado['accion_recomendada_para_el_gestor']}
                          </div>
                          <p style='color:#9b9b97;font-size:12px;margin-top:24px'>PostSale Gmail Integration · Análisis automático</p>
                        </div>""",
                    })
                except Exception as e:
                    log.warning(f"Error enviando alerta Gmail: {e}")

        procesados += 1
        resultados.append({
            "cliente": cliente.nombre,
            "email": email_addr,
            "emails_analizados": len(emails),
            "nivel_riesgo": resultado["nivel_riesgo"],
            "probabilidad_churn_porcentaje": resultado["probabilidad_churn_porcentaje"],
            "score_confianza": resultado["score_confianza"],
        })

    log.info(f"Gmail análisis completado: {procesados} clientes, {omitidos} omitidos")

    return {
        "estado": "ok",
        "procesados": procesados,
        "omitidos": omitidos,
        "dias_analizados": dias,
        "resultados": sorted(
            resultados,
            key=lambda r: r["probabilidad_churn_porcentaje"],
            reverse=True,
        ),
    }
'''


if __name__ == "__main__":
    print("=" * 60)
    print("  PostSale Gmail Integration")
    print("=" * 60)
    print("""
INSTRUCCIONES DE INTEGRACIÓN EN api.py:
=========================================

1. Agregar imports al inicio de api.py:

   from gmail_integration import (
       get_google_flow, get_gmail_service, gmail_esta_autorizado,
       obtener_emails_clientes, analizar_emails_cliente,
   )

2. Agregar variables de entorno:

   $env:GOOGLE_CLIENT_ID="tu-client-id"
   $env:GOOGLE_CLIENT_SECRET="tu-client-secret"

3. Copiar los 4 endpoints del string ENDPOINTS_GMAIL
   al final de api.py (antes del health check GET /)

4. Agregar a requirements.txt:
   google-auth
   google-auth-oauthlib
   google-auth-httplib2
   google-api-python-client

5. Agregar a .gitignore:
   gmail_token.json

6. Probar:
   uvicorn api:app --reload
   Ir a: http://localhost:8000/gmail/autorizar
    """)
    print("=" * 60)