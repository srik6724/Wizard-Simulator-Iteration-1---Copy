import javax.swing.*;
import javax.swing.Timer;

import java.awt.*;
import java.awt.event.*;
import java.awt.geom.*;
import java.util.*;
import java.util.List;

public class SpellSwingDemo extends JFrame {
    public static void main(String[] args) {
        SwingUtilities.invokeLater(SpellSwingDemo::new);
    }
    public SpellSwingDemo() {
        setTitle("Spell Animation — Swing/Java2D");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setContentPane(new SpellPanel());
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    // ---------------- PANEL ----------------
    static class SpellPanel extends JPanel implements ActionListener, MouseListener, KeyListener {
        // Size
        final int W = 960, H = 540;

        // Timing
        final Timer timer = new Timer(16, this); // ~60 FPS
        long last = System.nanoTime();
        boolean slowMo = false;

        // Scene actors
        final java.util.List<Actor> actors = new ArrayList<>();
        final Random rng = new Random();

        // Camera shake
        float camX = 0, camY = 0, shakeMag = 0;
        long shakeUntilMs = 0;

        // Colors/hues (we’ll vary hue for style, but Swing uses RGB — we’ll convert)
        float baseHue = 0.58f; // ~icy blue (0.0=red, 0.33=green, 0.66=blue, 1.0 wraps)
        final Point2D.Float caster = new Point2D.Float(W*0.20f, H*0.70f);

        SpellPanel() {
            setPreferredSize(new Dimension(W, H));
            setBackground(new Color(0x0A0F1A));
            setFocusable(true);
            addMouseListener(this);
            addKeyListener(this);
            timer.start();
            // demo fire once
            SwingUtilities.invokeLater(() -> castSpell(W*0.7f, H*0.4f));
        }

        // ------------- Loop -------------
        @Override public void actionPerformed(ActionEvent e) {
            long now = System.nanoTime();
            float dt = Math.min(0.033f, (now - last) / 1_000_000_000f);
            if (slowMo) dt *= 0.35f;
            last = now;

            updateCamera();
            stepActors(dt);
            repaint();
        }

        // ------------- Update -------------
        void stepActors(float dt) {
            for (int i = actors.size() - 1; i >= 0; i--) {
                Actor a = actors.get(i);
                if (!a.step(dt)) actors.remove(i);
            }
        }

        void updateCamera() {
            long nowMs = System.currentTimeMillis();
            float m = 0;
            if (nowMs < shakeUntilMs) {
                float k = Math.max(0f, Math.min(1f, (shakeUntilMs - nowMs) / 450f));
                m = shakeMag * k * (0.5f + 0.5f * (float)Math.sin(nowMs * 0.05f));
            }
            camX = (rng.nextFloat() * 2 - 1) * m;
            camY = (rng.nextFloat() * 2 - 1) * m;
        }

        void addShake(float power, int ms) {
            shakeMag = Math.max(shakeMag, power);
            shakeUntilMs = Math.max(shakeUntilMs, System.currentTimeMillis() + ms);
        }

        // ------------- Painting -------------
        @Override protected void paintComponent(Graphics gRaw) {
            super.paintComponent(gRaw);
            Graphics2D g = (Graphics2D) gRaw.create();
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Ground gradient
            Paint bg = new GradientPaint(0, 0, new Color(0x0A0F1A),
                    0, H, new Color(0x0B1220));
            g.setPaint(bg);
            g.fillRect(0, 0, W, H);

            // Ground rings
            g.setComposite(AlphaComposite.SrcOver.derive(0.35f));
            g.setColor(new Color(130, 160, 230, 180));
            g.translate(W/2.0 + camX*0.25, H*0.82 + camY*0.25);
            for (int i=0;i<6;i++) {
                float r = (i+1)*90f;
                g.setStroke(new BasicStroke(2f));
                g.draw(new Ellipse2D.Float(-r, -r, r*2, r*2));
            }
            g.translate(-(W/2.0 + camX*0.25), -(H*0.82 + camY*0.25));
            g.setComposite(AlphaComposite.SrcOver);

            // Camera transform
            g.translate(camX, camY);

            // Draw actors
            for (Actor a : actors) a.draw(g);

            // Soft vignette
            g.setComposite(AlphaComposite.SrcOver);
            Paint vign = new RadialGradientPaint(new Point2D.Float(W/2f, H/2f),
                    Math.max(W, H)*0.75f,
                    new float[]{0f, 1f},
                    new Color[]{new Color(0,0,0,0), new Color(0,0,0,90)});
            g.setPaint(vign);
            g.fillRect(0, 0, W, H);

            g.dispose();
            Toolkit.getDefaultToolkit().sync();
        }

        // ------------- Input -------------
        @Override public void mousePressed(MouseEvent e) {
            requestFocusInWindow();
            castSpell(e.getX(), e.getY());
        }
        @Override public void keyPressed(KeyEvent e) {
            if (e.getKeyCode() == KeyEvent.VK_SPACE) slowMo = !slowMo;
            if (e.getKeyCode() == KeyEvent.VK_R) { shakeMag = 0; shakeUntilMs = 0; }
        }
        @Override public void mouseReleased(MouseEvent e) {}
        @Override public void mouseClicked(MouseEvent e) {}
        @Override public void mouseEntered(MouseEvent e) {}
        @Override public void mouseExited(MouseEvent e) {}
        @Override public void keyTyped(KeyEvent e) {}
        @Override public void keyReleased(KeyEvent e) {}

        // ------------- Spell logic -------------
        void castSpell(float x, float y) {
            // small hue variance per cast
            float hue = baseHue + (rng.nextFloat()*0.05f - 0.025f);
            // Rune charge
            actors.add(new Rune(caster.x, caster.y, hue));
            // Projectile after slight delay
            final float tx = x, ty = y;
            new javax.swing.Timer(200, ev -> {
                ((Timer)ev.getSource()).stop();
                Projectile p = new Projectile(caster.x, caster.y, tx, ty, hue);
                actors.add(p);

                // Emit sparkle trail
                Timer trail = new Timer(25, e -> {
                    if (p.t >= 1f) { ((Timer)e.getSource()).stop(); return; }
                    for (int i=0;i<4;i++) {
                        float ang = rng.nextFloat() * (float)(Math.PI*2);
                        float sp = 30 + rng.nextFloat()*80;
                        Color c = Color.getHSBColor(hue, 1f, 1f);
                        Particle part = Particle.glow(p.x, p.y,
                                (float)Math.cos(ang)*sp, (float)Math.sin(ang)*sp,
                                0.45f + rng.nextFloat()*0.35f,
                                220 + rng.nextInt(280),
                                5f + rng.nextFloat()*5f,
                                c);
                        actors.add(part);
                    }
                });
                trail.start();

                // Watch for arrival
                Timer watch = new Timer(16, e -> {
                    if (p.t >= 1f) {
                        ((Timer)e.getSource()).stop();
                        impact(tx, ty, hue);
                    }
                });
                watch.start();
            }).start();
        }

        void impact(float x, float y, float hue) {
            actors.add(new Shockwave(x, y, hue));
            addShake(18f, 450);

            Color main = Color.getHSBColor(hue, 1f, 1f);
            // Burst
            for (int i=0;i<80;i++) {
                float ang = rng.nextFloat() * (float)(Math.PI*2);
                float sp = 120 + rng.nextFloat()*220;
                actors.add(Particle.glow(x, y,
                        (float)Math.cos(ang)*sp, (float)Math.sin(ang)*sp,
                        0.8f, 0, 7f + rng.nextFloat()*6f, main));
            }
            // Embers
            for (int i=0;i<50;i++) {
                float ang = rng.nextFloat() * (float)(Math.PI*2);
                float sp = 60 + rng.nextFloat()*120;
                Particle e = Particle.glow(x, y,
                        (float)Math.cos(ang)*sp, (float)Math.sin(ang)*sp,
                        1.3f, 0, 3f + rng.nextFloat()*3f,
                        Color.getHSBColor(hue+0.03f, 1f, 0.9f));
                e.gravity = 140f; e.drag = 0.96f;
                actors.add(e);
            }
        }

        // ------------- Helpers -------------
        static float lerp(float a, float b, float t){ return a + (b - a) * t; }
        static float easeOutExpo(float t){ return t==1f?1f:(float)(1 - Math.pow(2, -10 * t)); }
        static float clamp(float x, float a, float b){ return Math.max(a, Math.min(b, x)); }

        // ------------- Base Actor -------------
        interface Actor {
            boolean step(float dt); // dt in seconds
            void draw(Graphics2D g);
        }

        // ------------- Rune (charge circle) -------------
        class Rune implements Actor {
            final float x, y, hue;
            float t = 0f; // 0..1
            Rune(float x, float y, float hue){ this.x=x; this.y=y; this.hue=hue; }
            @Override public boolean step(float dt) {
                t += dt / 0.9f;
                return t < 1f;
            }
            @Override public void draw(Graphics2D g) {
                float k = clamp(t, 0f, 1f);
                float r = 60 + easeOutExpo(k)*40;
                Color c = Color.getHSBColor(hue, 1f, 1f);
                Stroke old = g.getStroke();
                g.setComposite(AlphaComposite.SrcOver.derive(1f - k));
                g.setColor(new Color(c.getRed(), c.getGreen(), c.getBlue(), 220));
                g.setStroke(new BasicStroke(3f));
                g.translate(x, y);
                g.rotate(System.currentTimeMillis()*0.002);
                // outer
                g.draw(new Ellipse2D.Float(-r, -r, r*2, r*2));
                // spokes
                int spokes = 8;
                for (int i=0;i<spokes;i++){
                    double a = i * (Math.PI*2) / spokes;
                    float len = (float)(r * (0.4 + 0.6 * k));
                    float sx = (float)Math.cos(a) * r * 0.4f;
                    float sy = (float)Math.sin(a) * r * 0.4f;
                    float ex = (float)Math.cos(a) * len;
                    float ey = (float)Math.sin(a) * len;
                    g.draw(new Line2D.Float(sx, sy, ex, ey));
                }
                g.rotate(-System.currentTimeMillis()*0.002);
                g.translate(-x, -y);
                g.setStroke(old);
                g.setComposite(AlphaComposite.SrcOver);
            }
        }

        // ------------- Projectile (quadratic bezier) -------------
        class Projectile implements Actor {
            final float x0,y0,x1,y1,hue;
            final Point2D.Float ctrl;
            float x, y, t = 0f;
            float life = 0.75f; // seconds
            final Deque<Point2D.Float> trail = new ArrayDeque<>();

            Projectile(float x0, float y0, float x1, float y1, float hue) {
                this.x0=x0; this.y0=y0; this.x1=x1; this.y1=y1; this.hue=hue;
                float midx = (x0+x1)/2f + (rng.nextFloat()*120f - 60f);
                float midy = (y0+y1)/2f - (80f + rng.nextFloat()*160f);
                this.ctrl = new Point2D.Float(midx, midy);
                this.x = x0; this.y = y0;
            }
            @Override public boolean step(float dt) {
                t += dt / life;
                t = clamp(t, 0f, 1f);
                // quadratic bezier: lerp( lerp(A,Ctrl,t), lerp(Ctrl,B,t), t )
                float ax = lerp(x0, ctrl.x, t);
                float ay = lerp(y0, ctrl.y, t);
                float bx = lerp(ctrl.x, x1, t);
                float by = lerp(ctrl.y, y1, t);
                x = lerp(ax, bx, t);
                y = lerp(ay, by, t);

                trail.addLast(new Point2D.Float(x, y));
                while (trail.size() > 50) trail.removeFirst();
                return t < 1f;
            }
            @Override public void draw(Graphics2D g) {
                // trail glow (additive-ish by layering)
                Color c = Color.getHSBColor(hue, 1f, 1f);
                int steps = 0;
                for (Point2D.Float p : trail) {
                    float age = (trail.size() - (steps++)) / 50f;
                    float a = (1f - age) * 0.7f;
                    float s = (1f - age) * 16f;
                    drawGlow(g, p.x, p.y, s*3, new Color(c.getRed(), c.getGreen(), c.getBlue(), (int)(255*a)));
                }
                // core orb
                drawGlow(g, x, y, 30, new Color(255,255,255,220));
                drawGlow(g, x, y, 60, new Color(c.getRed(), c.getGreen(), c.getBlue(), 200));
            }
        }

        // ------------- Shockwave ring -------------
        class Shockwave implements Actor {
            final float x,y,hue; float t=0f;
            Shockwave(float x,float y,float hue){ this.x=x; this.y=y; this.hue=hue; }
            @Override public boolean step(float dt) {
                t += dt / 0.65f;
                return t < 1f;
            }
            @Override public void draw(Graphics2D g) {
                float r = easeOutExpo(t) * Math.max(W, H) * 0.3f;
                Color c = Color.getHSBColor(hue, 1f, 1f);
                g.setComposite(AlphaComposite.SrcOver.derive(1f - t));
                g.setColor(new Color(c.getRed(), c.getGreen(), c.getBlue(), 200));
                g.setStroke(new BasicStroke(lerp(8f, 1f, t)));
                g.draw(new Ellipse2D.Float(x-r, y-r, r*2, r*2));
                g.setComposite(AlphaComposite.SrcOver);
            }
        }

        // ------------- Particle -------------
        static class Particle implements Actor {
            float x,y, vx,vy, life, maxLife, size;
            float drag=0.97f, gravity=0f; // px/s²
            Color color;

            static Particle glow(float x,float y,float vx,float vy,float lifeSec,int lifeJitterMs,float size,Color c){
                Particle p = new Particle();
                p.x=x; p.y=y; p.vx=vx; p.vy=vy;
                p.maxLife = lifeSec + (lifeJitterMs>0 ? (new Random()).nextInt(lifeJitterMs)/1000f : 0f);
                p.life = p.maxLife; p.size = size; p.color = c;
                return p;
            }

            @Override public boolean step(float dt) {
                vx *= drag;
                vy = vy * drag + gravity * dt;
                x += vx * dt;
                y += vy * dt;
                size *= 0.996f;
                life -= dt;
                return life > 0 && size > 0.5f;
            }
            @Override public void draw(Graphics2D g) {
                float t = Math.max(0f, life / Math.max(0.0001f, maxLife));
                Color c = new Color(color.getRed(), color.getGreen(), color.getBlue(), (int)(220 * t));
                drawGlow(g, x, y, size*3f, c);
            }
        }

        // ------------- Glow helper -------------
        static void drawGlow(Graphics2D g, float x, float y, float radius, Color c) {
            float r = Math.max(1f, radius);
            Point2D center = new Point2D.Float(x, y);
            float[] dist = {0f, 1f};
            Color inner = new Color(c.getRed(), c.getGreen(), c.getBlue(), Math.min(255, c.getAlpha()));
            Color outer = new Color(c.getRed(), c.getGreen(), c.getBlue(), 0);
            RadialGradientPaint p = new RadialGradientPaint(center, r, dist, new Color[]{inner, outer});
            Paint old = g.getPaint();
            g.setPaint(p);
            g.fill(new Ellipse2D.Float(x - r, y - r, r*2, r*2));
            g.setPaint(old);
        }
    }
}

