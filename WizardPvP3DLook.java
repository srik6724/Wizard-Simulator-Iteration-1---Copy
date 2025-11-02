import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.*;
import java.util.Random;

/**
 * WizardPvP3DLook
 * - Pure Swing/Java2D (no external libs)
 * - Fake 3D via perspective scaling, gradients, and shadows
 * - Power-Nova style spell: wand raise -> projectile arc -> center glyph + pillar + shockwave -> damage
 */
public class WizardPvP3DLook extends JPanel implements ActionListener {
    // Timing / animation
    private final Timer timer = new Timer(16, this); // ~60 FPS
    private long t0 = System.currentTimeMillis();
    private int frame = 0;
    private boolean casting = false;

    // Camera-ish params
    private double camHeight = 380;       // how “high” camera is
    private double camTilt = Math.toRadians(58); // tilt toward floor
    private double arenaRadius = 250;

    // Wizards
    private Point2D.Double leftPos  = new Point2D.Double(-180,  60);
    private Point2D.Double rightPos = new Point2D.Double( 180, -60);
    private int maxHP = 1000;
    private int hpL = maxHP;
    private int hpR = maxHP;

    // Spell state
    private enum Phase { IDLE, WAND_UP, PROJECTILE, IMPACT, COOL }
    private Phase phase = Phase.IDLE;
    private long phaseStart = 0;
    private int dmgOnHit = 260;
    private Random rng = new Random();

    // Projectile path cache
    private CubicCurve2D.Double projCurve = new CubicCurve2D.Double();
    private Point2D.Double wandTipWorld = new Point2D.Double();

    public WizardPvP3DLook() {
        setBackground(new Color(0x0b0d13));
        setPreferredSize(new Dimension(1000, 650));

        // Click or spacebar to cast
        addMouseListener(new MouseAdapter() {
            @Override public void mouseClicked(MouseEvent e) { startCastIfPossible(); }
        });
        setFocusable(true);
        addKeyListener(new KeyAdapter() {
            @Override public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_SPACE) startCastIfPossible();
            }
        });

        timer.start();
    }

    private void startCastIfPossible() {
        if (casting) return;
        casting = true;
        phase = Phase.WAND_UP;
        phaseStart = nowMs();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        final int W = getWidth(), H = getHeight();
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Camera "origin" (screen center) and floor projection helpers
        Point center = new Point(W/2, (int)(H*0.62));

        // Background vignette
        Paint sky = new RadialGradientPaint(center, (float)(W*0.75),
                new float[]{0f, 1f},
                new Color[]{new Color(0x111625), new Color(0x07090f)});
        g2.setPaint(sky);
        g2.fillRect(0,0,W,H);

        // Ambient haze
        g2.setPaint(new Color(0x0f1424));
        g2.fill(new Rectangle2D.Double(0, H*0.62, W, H));

        // Light
        g2.setPaint(new Color(255,255,255,18));
        g2.fill(new Ellipse2D.Double(center.x-200, center.y-180, 400, 280));

        // Floor disc (ellipse for perspective)
        drawArena(g2, center);

        // Idle bob
        double bob = Math.sin((nowMs()-t0)*0.002) * 6.0;

        // Draw wizards with fake depth: further (lower y) → slightly smaller scale
        drawWizard(g2, center, leftPos,  new Color(0x5ea8ff), 1.0 + bob*0.002, true);
        drawWizard(g2, center, rightPos, new Color(0xff6961), 0.96 - bob*0.002, false);

        // Center sigil / shockwave / pillar (only during impact)
        drawImpactFX(g2, center);

        // Projectile (during projectile phase)
        drawProjectile(g2, center);

        // UI: health bars + hint
        drawUI(g2, W, H);

        g2.dispose();
    }

    private void drawArena(Graphics2D g2, Point center) {
        double ry = arenaRadius * 0.35;               // vertical radius (squash for perspective)
        Shape floor = new Ellipse2D.Double(center.x - arenaRadius, center.y - ry, arenaRadius*2, ry*2);

        // ring edge
        g2.setStroke(new BasicStroke(2f));
        g2.setPaint(new Color(0x101523));
        g2.draw(floor);

        // subtle fill
        g2.setPaint(new Color(0x0f1424));
        g2.fill(floor);

        // glowing runes
        g2.setStroke(new BasicStroke(1.5f));
        g2.setPaint(new Color(0x567bff));
        for (int i = 0; i < 6; i++) {
            double r = 150 + i*18;
            Shape ring = new Ellipse2D.Double(center.x - r, center.y - r*0.35, 2*r, 2*r*0.35);
            g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.10f));
            g2.draw(ring);
        }
        g2.setComposite(AlphaComposite.SrcOver);
    }

    private void drawWizard(Graphics2D g2, Point center, Point2D.Double pos, Color robe, double idleScale, boolean isLeft) {
        // World→screen: treat pos as floor space; convert to ellipse space
        Point2D.Double s = floorToScreen(center, pos.x, pos.y);
        double depthScale = map(pos.y, -arenaRadius, arenaRadius, 0.90, 1.08); // front = bigger
        double scale = depthScale * idleScale;

        // Shadow
        Shape shadow = new Ellipse2D.Double(s.x - 25*scale, s.y - 5*scale, 50*scale, 18*scale);
        g2.setPaint(new Color(0,0,0,70));
        g2.fill(shadow);

        AffineTransform at = g2.getTransform();
        g2.translate(s.x, s.y - 60*scale);
        g2.scale(scale, scale);

        // Robe (cone-like: triangle with curved base)
        Path2D robeShape = new Path2D.Double();
        robeShape.moveTo(-30, 60);
        robeShape.curveTo(-18, 20, -12, -8, 0, -12);
        robeShape.curveTo(12, -8, 18, 20, 30, 60);
        robeShape.closePath();

        // Robe paint (vertical gradient)
        GradientPaint robeGrad = new GradientPaint(0f, -20f, robe.brighter(), 0f, 70f, robe.darker());
        g2.setPaint(robeGrad);
        g2.fill(robeShape);

        // Torso tube
        Shape torso = new RoundRectangle2D.Double(-14, -36, 28, 36, 10, 10);
        g2.setPaint(new Color(0x2b3a57));
        g2.fill(torso);

        // Head
        g2.setPaint(new Color(0xffe0bd));
        g2.fill(new Ellipse2D.Double(-10, -56, 20, 20));

        // Hat
        g2.setPaint(new Color(0x1c2540));
        Polygon hat = new Polygon(new int[]{0, -16, 16}, new int[]{-78, -48, -48}, 3);
        g2.fill(hat);

        // Wand arm
        g2.setPaint(new Color(0x2b3a57));
        Shape arm = new RoundRectangle2D.Double(8, -32, 8, 24, 8, 8);
        g2.rotate(isLeft ? -Math.toRadians(24) : Math.toRadians(24));
        g2.fill(arm);

        // Wand
        g2.setPaint(new Color(0x7a5a2b));
        Shape wand = new RoundRectangle2D.Double(12, -46, 4, 40, 3, 3);
        g2.fill(wand);

        // Wand tip glow (small circle)
        g2.setPaint(new Color(0x9fd6ff));
        g2.fill(new Ellipse2D.Double(12, -54, 7, 7));

        // Cache wand-tip world (for projectile start)
        Point2D tipLocal = new Point2D.Double(15, -50);
        Point2D tipRot = rotatePoint(tipLocal, isLeft ? -24 : 24);
        Point2D tipWorld = new Point2D.Double(s.x + tipRot.getX()*scale, s.y - 60*scale + tipRot.getY()*scale);
        if (isLeft) {
            wandTipWorld.setLocation(tipWorld);
        }

        g2.setTransform(at);
    }

    private void drawProjectile(Graphics2D g2, Point center) {
        if (phase != Phase.PROJECTILE) return;

        long elapsed = nowMs() - phaseStart;
        double t = clamp(elapsed / 450.0, 0, 1);
        t = easeInOutQuad(t);

        // Build curve from wandTip → center, with upward arc
        Point2D.Double C = floorToScreen(center, 0, 0.0);
        double ctrlLift = -120; // arc height

        projCurve.setCurve(wandTipWorld.x, wandTipWorld.y,
                lerp(wandTipWorld.x, C.x, 0.33), lerp(wandTipWorld.y, C.y, 0.33) + ctrlLift,
                lerp(wandTipWorld.x, C.x, 0.66), lerp(wandTipWorld.y, C.y, 0.66) + ctrlLift,
                C.x, C.y);

        Point2D.Double p = pointOnCubic(projCurve, t);

        // glow trail
        g2.setStroke(new BasicStroke(2.2f));
        g2.setPaint(new Color(0x5ab8ff));
        drawCubicPartial(g2, projCurve, t);

        // projectile orb
        float alpha = 0.8f;
        g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, alpha));
        g2.setPaint(new Color(0x9fd6ff));
        g2.fill(new Ellipse2D.Double(p.x-6, p.y-6, 12, 12));

        if (t >= 1.0 - 1e-6) {
            // impact
            phase = Phase.IMPACT;
            phaseStart = nowMs();
            hpR = Math.max(0, hpR - dmgOnHit);
        }
        g2.setComposite(AlphaComposite.SrcOver);
    }

    private void drawImpactFX(Graphics2D g2, Point center) {
        if (phase != Phase.IMPACT) return;

        long elapsed = nowMs() - phaseStart;
        double t = clamp(elapsed / 850.0, 0, 1);

        // Center point on floor
        Point2D.Double C = floorToScreen(center, 0, 0.0);

        // 1) ground glyph expansion
        double baseR = 28;
        double R = baseR + t * 160;
        Shape ring = new Ellipse2D.Double(C.x - R, C.y - R*0.35, 2*R, 2*R*0.35);

        g2.setStroke(new BasicStroke(2.5f));
        g2.setPaint(new Color(0x87b3ff));
        g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, (float)(0.85*(1-t))));
        g2.draw(ring);

        // 2) pillar
        double tall = 200 * (1 - Math.pow((t-0.1)*1.2, 2));
        tall = Math.max(0, tall);
        Shape pillar = new RoundRectangle2D.Double(C.x - 10, C.y - 90 - tall*0.3, 20, 90 + tall, 20, 20);
        g2.setPaint(new Color(0x9ad0ff));
        g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, (float)(0.6*(1-t))));
        g2.fill(pillar);

        // 3) shockwave (thicker ring)
        double SW = 40 + t * 220;
        Stroke swStroke = new BasicStroke((float)(12*(1-t)+2));
        g2.setStroke(swStroke);
        g2.setPaint(new Color(0xcbe2ff));
        Shape sw = new Ellipse2D.Double(C.x - SW, C.y - SW*0.35, 2*SW, 2*SW*0.35);
        g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, (float)(0.7*(1-t))));
        g2.draw(sw);

        // End
        if (t >= 1.0 - 1e-6) {
            phase = Phase.COOL;
            phaseStart = nowMs();
        }
        g2.setComposite(AlphaComposite.SrcOver);
    }

    private void drawUI(Graphics2D g2, int W, int H) {
        // Health bars
        int bw = (int)(W * 0.36);
        int bh = 16;
        int pad = 14;

        // Left
        drawBar(g2, pad, pad, bw, bh, "Caster", hpL, maxHP, true);
        // Right
        drawBar(g2, W - pad - bw, pad, bw, bh, "Opponent", hpR, maxHP, false);

        // Hint
        g2.setPaint(new Color(0xa8b6ff));
        g2.setFont(g2.getFont().deriveFont(Font.PLAIN, 14f));
        String hint = "Click or press Space to cast Power Nova ✨";
        g2.drawString(hint, pad, H - pad);
    }

    private void drawBar(Graphics2D g2, int x, int y, int w, int h, String name, int hp, int max, boolean left) {
        g2.setPaint(new Color(0x1a2030));
        g2.fillRoundRect(x, y, w, h, h, h);
        g2.setPaint(new Color(0x28324a));
        g2.drawRoundRect(x, y, w, h, h, h);

        double pct = Math.max(0, Math.min(1, hp / (double)max));
        GradientPaint gp = new GradientPaint(x, y, new Color(0x54ffa8), x+w, y, new Color(0x2ee5ff));
        g2.setPaint(gp);
        g2.fillRoundRect(x, y, (int)(w*pct), h, h, h);

        g2.setPaint(new Color(0xe6edff));
        g2.setFont(g2.getFont().deriveFont(Font.BOLD, 13f));
        if (left) g2.drawString(name, x, y-5);
        else {
            FontMetrics fm = g2.getFontMetrics();
            g2.drawString(name, x + w - fm.stringWidth(name), y-5);
        }
    }

    // ===== Animation driver =====
    @Override public void actionPerformed(ActionEvent e) {
        frame++;

        // simple state machine timeouts
        switch (phase) {
            case IDLE:
                break;
            case WAND_UP:
                if (nowMs() - phaseStart > 350) {
                    phase = Phase.PROJECTILE;
                    phaseStart = nowMs();
                }
                break;
            case PROJECTILE:
                // handled in drawProjectile; transition on arrival
                break;
            case IMPACT:
                // handled in drawImpactFX; transition on finish
                break;
            case COOL:
                if (nowMs() - phaseStart > 450) {
                    phase = Phase.IDLE;
                    casting = false;
                }
                break;
        }
        repaint();
    }

    // ===== Helpers =====

    private long nowMs() { return System.currentTimeMillis(); }

    private static double map(double v, double a, double b, double c, double d) {
        double t = (v - a) / (b - a);
        return c + (d - c) * t;
    }

    private static double clamp(double v, double a, double b) {
        return Math.max(a, Math.min(b, v));
    }

    private static double lerp(double a, double b, double t) {
        return a + (b - a) * t;
    }

    private static double easeInOutQuad(double t) {
        return (t < 0.5) ? (2*t*t) : (1 - Math.pow(-2*t + 2, 2)/2);
    }

    private Point2D.Double floorToScreen(Point center, double x, double y) {
        // Floor XY projected onto ellipse: squash Y
        double sx = center.x + x;
        double sy = center.y + y * 0.35;
        return new Point2D.Double(sx, sy);
    }

    private static Point2D rotatePoint(Point2D p, double deg) {
        double r = Math.toRadians(deg);
        double c = Math.cos(r), s = Math.sin(r);
        return new Point2D.Double(p.getX()*c - p.getY()*s, p.getX()*s + p.getY()*c);
    }

    // Draw a portion of a cubic curve (0..t)
    private static void drawCubicPartial(Graphics2D g2, CubicCurve2D.Double c, double t) {
        Path2D path = new Path2D.Double();
        path.moveTo(c.x1, c.y1);
        int steps = 32;
        for (int i=1; i<=steps; i++){
            double u = i/(double)steps;
            if (u > t) break;
            Point2D p = pointOnCubic(c, u);
            path.lineTo(p.getX(), p.getY());
        }
        g2.draw(path);
    }

    // Evaluate cubic Bezier at t
    private static Point2D.Double pointOnCubic(CubicCurve2D.Double c, double t) {
        double u = 1-t;
        double x = u*u*u*c.x1 + 3*u*u*t*c.ctrlx1 + 3*u*t*t*c.ctrlx2 + t*t*t*c.x2;
        double y = u*u*u*c.y1 + 3*u*u*t*c.ctrly1 + 3*u*t*t*c.ctrly2 + t*t*t*c.y2;
        return new Point2D.Double(x,y);
    }

    // ===== Main =====
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame f = new JFrame("Wizard PvP — Power Nova (Fake 3D, pure Swing)");
            f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            WizardPvP3DLook panel = new WizardPvP3DLook();
            f.setContentPane(panel);
            f.pack();
            f.setLocationRelativeTo(null);
            f.setVisible(true);
        });
    }
}

