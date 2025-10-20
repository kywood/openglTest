


# app.py
import math, os, ctypes, random
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import Response
from OpenGL.GL import *
import glfw
from PIL import Image
from typing import Tuple

app = FastAPI(title="OpenGL Offscreen Image Server")

# -------------------- OpenGL helpers --------------------
VSH = """
#version 330 core
layout (location=0) in vec2 aPos;
uniform mat4 uMVP;
void main(){ gl_Position = uMVP * vec4(aPos, 0.0, 1.0); }
"""
FSH = """
#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main(){ FragColor = vec4(uColor, 1.0); }
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
    # 비가시 창 (주: 리눅스 서버에서 X/Wayland 없으면 xvfb나 EGL/OSMesa 필요)
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)  # macOS
    win = glfw.create_window(w, h, "offscreen", None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("GLFW window create failed (headless needs X/Wayland/EGL)")
    glfw.make_context_current(win)
    return win

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
    arr = np.flipud(arr)  # 상하반전
    im = Image.fromarray(arr, mode="RGBA")
    out = bytes()
    import io
    bio = io.BytesIO()
    im.save(bio, format="PNG")
    return bio.getvalue()

# -------------------- Geometry builders (2D in NDC) --------------------
def circle_polyline(radius: float, segments: int = 512) -> np.ndarray:
    a = np.linspace(0, 2*math.pi, segments+1, dtype=np.float32)
    xy = np.stack([radius*np.cos(a), radius*np.sin(a)], axis=1)
    return xy.astype(np.float32)

def random_point_in_circle(radius: float) -> Tuple[float,float]:
    # 균일 분포: r = R*sqrt(u)
    u = random.random()
    r = radius * math.sqrt(u)
    th = random.random() * 2*math.pi
    return (r*math.cos(th), r*math.sin(th))

def arrow_geometry(p: Tuple[float,float], dir_rad: float,
                   shaft_len: float, head_len: float, head_w: float) -> Tuple[np.ndarray, np.ndarray]:
    """화살표: 샤프트(GL_LINES, 2점) + 머리(GL_TRIANGLES, 3점) 모두 NDC 좌표 반환"""
    px, py = p
    dx, dy = math.cos(dir_rad), math.sin(dir_rad)
    # line
    qx, qy = px + dx*shaft_len, py + dy*shaft_len
    shaft = np.array([[px, py], [qx, qy]], dtype=np.float32)
    # head triangle
    nx, ny = -dy, dx
    tip = np.array([qx + dx*head_len, qy + dy*head_len], np.float32)
    base = np.array([qx - dx*head_len, qy - dy*head_len], np.float32)
    left  = base + np.array([nx, ny], np.float32)*head_w
    right = base - np.array([nx, ny], np.float32)*head_w
    head = np.vstack([tip, left, right]).astype(np.float32)
    return shaft, head

# -------------------- Renderer --------------------
def render_image(width: int, height: int, arrows: int = 100) -> bytes:
    win = create_offscreen_context(width, height)
    try:
        fbo, tex, rbo = create_fbo(width, height)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glViewport(0, 0, width, height)

        prog = compile_program(VSH, FSH)
        uMVP = glGetUniformLocation(prog, "uMVP")
        uClr = glGetUniformLocation(prog, "uColor")

        # 배경 클리어
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.08, 0.09, 0.11, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

        # 화면비 유지 ortho
        aspect = width/float(height)
        L,R,B,T = -1.0, 1.0, -1.0, 1.0
        if aspect >= 1.0:
            L, R = -aspect, aspect
        else:
            B, T = -1.0/aspect, 1.0/aspect
        M = ortho(L,R,B,T)

        # 원(테두리)
        radius = 0.8  # NDC 기준 큰 원
        circle = circle_polyline(radius, 512)

        vao_circle = glGenVertexArrays(1); glBindVertexArray(vao_circle)
        vbo_circle = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo_circle)
        glBufferData(GL_ARRAY_BUFFER, circle.nbytes, circle, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

        glUseProgram(prog)
        glUniformMatrix4fv(uMVP, 1, GL_FALSE, M)
        glBindVertexArray(vao_circle)
        glUniform3f(uClr, 0.92, 0.95, 0.98)
        glLineWidth(1.0)
        glDrawArrays(GL_LINE_STRIP, 0, circle.shape[0])

        # 임의 화살표들
        shafts = []
        heads  = []
        for _ in range(arrows):
            # 원 내부에서 시작점 선택, 방향 임의
            px, py = random_point_in_circle(radius*0.9)
            dir_rad = random.random()*2*math.pi
            shaft_len = 0.08 + random.random()*0.10   # 0.08~0.18
            head_len  = shaft_len*0.45
            head_w    = head_len*0.3
            shaft, head = arrow_geometry((px,py), dir_rad, shaft_len, head_len, head_w)
            shafts.append(shaft); heads.append(head)

        shafts_np = np.vstack(shafts).astype(np.float32) if shafts else np.zeros((0,2), np.float32)
        heads_np  = np.vstack(heads).astype(np.float32)   if heads  else np.zeros((0,2), np.float32)

        # 업로드 & 그리기 (샤프트)
        vao_s = glGenVertexArrays(1); glBindVertexArray(vao_s)
        vbo_s = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo_s)
        glBufferData(GL_ARRAY_BUFFER, shafts_np.nbytes, shafts_np, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glUniform3f(uClr, 0.70, 0.84, 1.00)
        glLineWidth(1.0)
        glDrawArrays(GL_LINES, 0, shafts_np.shape[0])

        # 업로드 & 그리기 (머리)
        vao_h = glGenVertexArrays(1); glBindVertexArray(vao_h)
        vbo_h = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo_h)
        glBufferData(GL_ARRAY_BUFFER, heads_np.nbytes, heads_np, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glUniform3f(uClr, 0.98, 0.76, 0.18)
        glDrawArrays(GL_TRIANGLES, 0, heads_np.shape[0])

        # 픽셀 추출
        png = read_pixels_png_bytes(width, height)

        # 정리
        for vao,vbo in [(vao_circle, vbo_circle), (vao_s, vbo_s), (vao_h, vbo_h)]:
            glDeleteBuffers(1, [vbo]); glDeleteVertexArrays(1, [vao])
        glDeleteProgram(prog)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteRenderbuffers(1, [rbo]); glDeleteTextures(1, [tex]); glDeleteFramebuffers(1, [fbo])

        return png
    finally:
        destroy_context(win)

# -------------------- FastAPI endpoint --------------------
@app.get("/render")
def render(
    width: int = Query(..., gt=0, le=8192, description="이미지 가로(px)"),
    height: int = Query(..., gt=0, le=8192, description="이미지 세로(px)"),
    arrows: int = Query(100, gt=0, le=10000, description="화살표 개수"),
):
    try:
        png = render_image(width, height, arrows)
        return Response(content=png, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 선택: 직접 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gl1:app", host="0.0.0.0", port=8000)
