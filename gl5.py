# gl1.py  — FastAPI + PyOpenGL + GLFW (오프스크린)
import math, ctypes
from typing import Optional

import numpy as np
from typing import Tuple
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import Response
from OpenGL.GL import *
import glfw
from PIL import Image

app = FastAPI(title="OpenGL Offscreen Image Server (Vectorized + Random Colors)")

# -------------------- Shaders --------------------
VSH = """
#version 330 core
layout (location=0) in vec2 aPos;
layout (location=1) in vec3 aColor;   // per-vertex color
out vec3 vColor;
uniform mat4 uMVP;
void main(){
    vColor = aColor;
    gl_Position = uMVP * vec4(aPos, 0.0, 1.0);
}
"""
FSH = """
#version 330 core
in vec3 vColor;
out vec4 FragColor;
uniform vec3 uColor;   // for circle only
void main(){
    // If a vertex color is provided, use it; (circle uses uColor via setting attrib to 0)
    FragColor = vec4(vColor, 1.0);
}
"""

def compile_program(vsrc, fsrc) -> int:
    def _compile(kt, src):
        sid = glCreateShader(kt)
        glShaderSource(sid, src)
        glCompileShader(sid)
        if not glGetShaderiv(sid, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(sid).decode())
        return sid
    v = _compile(GL_VERTEX_SHADER, vsrc)
    f = _compile(GL_FRAGMENT_SHADER, fsrc)
    p = glCreateProgram()
    glAttachShader(p, v); glAttachShader(p, f)
    glLinkProgram(p)
    if not glGetProgramiv(p, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(p).decode())
    glDeleteShader(v); glDeleteShader(f)
    return p

def ortho(l,r,b,t,n=-1.0,f=1.0):
    M = np.eye(4, dtype=np.float32)
    M[0,0]=2/(r-l); M[1,1]=2/(t-b); M[2,2]=-2/(f-n)
    M[3,0]=-(r+l)/(r-l); M[3,1]=-(t+b)/(t-b); M[3,2]=-(f+n)/(f-n)
    return M

def create_offscreen_context(w: int, h: int):
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # 핵심: EGL API를 사용해 GPU 컨텍스트 생성
    # glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)

    win = glfw.create_window(w, h, "offscreen", None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("EGL context creation failed — driver not available")
    glfw.make_context_current(win)
    return win
#
# def create_offscreen_context(w: int, h: int):
#     if not glfw.init():
#         raise RuntimeError("GLFW init failed")
#     glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
#     glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
#     glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
#     glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
#     glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)  # macOS
#     win = glfw.create_window(w, h, "offscreen", None, None)
#     if not win:
#         glfw.terminate()
#         raise RuntimeError("GLFW window create failed (headless needs X/Wayland/EGL)")
#     glfw.make_context_current(win)
#     return win

def destroy_context(win):
    glfw.destroy_window(win)
    glfw.terminate()

def create_fbo(w, h) -> Tuple[int,int,int]:
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h)

    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo)
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("FBO incomplete")
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return fbo, tex, rbo

def read_pixels_png_bytes(w, h) -> bytes:
    buf = glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    arr = np.flipud(arr)
    import io
    bio = io.BytesIO()
    Image.fromarray(arr, mode="RGBA").save(bio, format="PNG")
    return bio.getvalue()

def circle_polyline(radius: float, segments: int = 512) -> np.ndarray:
    a = np.linspace(0, 2*math.pi, segments+1, dtype=np.float32)
    xy = np.stack([radius*np.cos(a), radius*np.sin(a)], axis=1)
    return xy.astype(np.float32)

def set_safe_line_width(requested: float) -> float:
    rng = (GLfloat * 2)()
    glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, rng)
    min_w, max_w = float(rng[0]), float(rng[1])
    safe = max(min_w, min(requested, max_w))
    glLineWidth(safe)
    return safe

# -------------------- Renderer (vectorized + per-vertex color) --------------------
def render_image(width: int, height: int, arrows: int = 100, seed: Optional[int] = None) -> bytes:
# def render_image(width: int, height: int, arrows: int = 100, seed: int | None = None) -> bytes:
    if seed is not None:
        np.random.seed(seed)

    win = create_offscreen_context(width, height)
    try:
        fbo, tex, rbo = create_fbo(width, height)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glViewport(0, 0, width, height)

        prog = compile_program(VSH, FSH)
        uMVP = glGetUniformLocation(prog, "uMVP")
        uClr = glGetUniformLocation(prog, "uColor")

        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.08, 0.09, 0.11, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

        # 화면비 유지 ortho
        aspect = width/float(height)
        L,R,B,T = -1.0, 1.0, -1.0, 1.0
        if aspect >= 1.0: L, R = -aspect, aspect
        else: B, T = -1.0/aspect, 1.0/aspect
        M = ortho(L,R,B,T)

        glUseProgram(prog)
        glUniformMatrix4fv(uMVP, 1, GL_FALSE, M)

        # ----- 원(테두리) : 단색 -----
        radius = 0.8
        circle = circle_polyline(radius, 512)

        vao_circle = glGenVertexArrays(1); glBindVertexArray(vao_circle)
        vbo_circle = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo_circle)
        glBufferData(GL_ARRAY_BUFFER, circle.nbytes, circle, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # circle은 vColor가 없으니 location=1 비활성화 (또는 1을 0으로 바인딩)
        glDisableVertexAttribArray(1)

        # 단색을 쓰려면 FSH를 건드리지 않고, vColor를 0으로 만들어도 되지만
        # 여기선 라인 두께만 맞춰서 단일 그리기
        set_safe_line_width(1.0)
        # 단색을 쓰려면 별도 셰이더가 필요하지만, 간단히 컬러 어트리뷰트를 0으로 바인딩하는 트릭:
        # glVertexAttrib3f(1, 0.92, 0.95, 0.98) 를 쓰려면 enabled 되어있지 않아야 함
        glVertexAttrib3f(1, 0.92, 0.95, 0.98)  # 고정값
        glDrawArrays(GL_LINE_STRIP, 0, circle.shape[0])

        # ----- 화살표: 위치/길이/색 모두 벡터화 -----
        N = int(arrows)

        theta = np.random.rand(N).astype(np.float32) * (2*np.pi)
        r = np.sqrt(np.random.rand(N).astype(np.float32)) * (radius * 0.9)
        px = r * np.cos(theta)
        py = r * np.sin(theta)

        dir_rad = np.random.rand(N).astype(np.float32) * (2*np.pi)
        dx = np.cos(dir_rad); dy = np.sin(dir_rad)

        # 길이 랜덤 (원하는 범위로 조절)
        shaft_len = 0.05 + np.random.rand(N).astype(np.float32)*0.15  # 0.05 ~ 0.20
        head_len  = shaft_len * (0.35 + 0.25*np.random.rand(N).astype(np.float32))  # 0.35~0.60 배
        head_w    = head_len * 0.1  # 얇은 머리

        # 샤프트 끝점
        qx = px + dx*shaft_len
        qy = py + dy*shaft_len

        # --- 버텍스 데이터 (샤프트: 2점/화살표, 머리: 3점/화살표) ---
        shafts = np.column_stack([px, py, qx, qy]).reshape(-1, 2).astype(np.float32)

        nx = -dy; ny = dx
        tipx = qx + dx*head_len; tipy = qy + dy*head_len
        basex = qx - dx*head_len; basey = qy - dy*head_len
        leftx  = basex + nx*head_w; lefty  = basey + ny*head_w
        rightx = basex - nx*head_w; righty = basey - ny*head_w
        heads = np.column_stack([tipx, tipy, leftx, lefty, rightx, righty]).reshape(-1,2).astype(np.float32)

        # --- 색상: 화살표별 랜덤 RGB, 버텍스 수에 맞춰 반복 ---
        # 샤프트는 화살표당 2버텍스, 머리는 3버텍스
        colors_arrow = np.random.rand(N, 3).astype(np.float32)  # (N,3) in [0,1)
        colors_shaft = np.repeat(colors_arrow, 2, axis=0)       # (2N,3)
        colors_head  = np.repeat(colors_arrow, 3, axis=0)       # (3N,3)

        # ----- 업로드 & 드로우: 샤프트 (positions + colors) -----
        vao_s = glGenVertexArrays(1); glBindVertexArray(vao_s)
        vbo_s_pos = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo_s_pos)
        glBufferData(GL_ARRAY_BUFFER, shafts.nbytes, shafts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        vbo_s_col = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo_s_col)
        glBufferData(GL_ARRAY_BUFFER, colors_shaft.nbytes, colors_shaft, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)

        set_safe_line_width(1.0)
        glDrawArrays(GL_LINES, 0, shafts.shape[0])

        # ----- 업로드 & 드로우: 머리 (positions + colors) -----
        vao_h = glGenVertexArrays(1); glBindVertexArray(vao_h)
        vbo_h_pos = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo_h_pos)
        glBufferData(GL_ARRAY_BUFFER, heads.nbytes, heads, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        vbo_h_col = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo_h_col)
        glBufferData(GL_ARRAY_BUFFER, colors_head.nbytes, colors_head, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)

        glDrawArrays(GL_TRIANGLES, 0, heads.shape[0])

        # ----- readback -----
        png = read_pixels_png_bytes(width, height)

        # 정리
        for vao, vbos in [
            (vao_circle, [vbo_circle]),
            (vao_s, [vbo_s_pos, vbo_s_col]),
            (vao_h, [vbo_h_pos, vbo_h_col]),
        ]:
            for v in vbos: glDeleteBuffers(1, [v])
            glDeleteVertexArrays(1, [vao])

        glDeleteProgram(prog)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteRenderbuffers(1, [rbo]); glDeleteTextures(1, [tex]); glDeleteFramebuffers(1, [fbo])

        return png
    finally:
        destroy_context(win)

# -------------------- FastAPI endpoint --------------------
@app.get("/render")
def render(
    width: int = Query(..., gt=1, le=8192, description="이미지 가로(px)"),
    height: int = Query(..., gt=1, le=8192, description="이미지 세로(px)"),
    arrows: int = Query(100, gt=1, le=300000, description="화살표 개수"),
    seed: Optional[int] = Query(None, description="난수 시드(재현성)"),
):
    try:
        png = render_image(width, height, arrows, seed)
        return Response(content=png, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 선택: 직접 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gl5:app", host="0.0.0.0", port=7000)
