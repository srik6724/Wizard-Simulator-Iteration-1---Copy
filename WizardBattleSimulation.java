import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.*;

public class WizardBattleSimulation extends JPanel implements ActionListener, KeyListener {
    private Timer timer;
    private int animationFrame = 0;
    private int casterHealth = 100;
    private int targetHealth = 100;
    private boolean isCasting = false;

    public WizardBattleSimulation() {
        timer = new Timer(50, this); // 50ms per frame
        timer.start();
        setBackground(new Color(10, 10, 30)); // Darker starry background
        setFocusable(true);
        addKeyListener(this);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Draw 3D arena (trapezoid with grid for perspective)
        GeneralPath arena = new GeneralPath();
        arena.moveTo(100, 450); // Bottom left
        arena.lineTo(500, 450); // Bottom right
        arena.lineTo(400, 150); // Top right
        arena.lineTo(200, 150); // Top left
        arena.closePath();
        GradientPaint arenaGradient = new GradientPaint(300, 150, new Color(120, 120, 150), 300, 450, new Color(40, 40, 60));
        g2d.setPaint(arenaGradient);
        g2d.fill(arena);
        // Grid lines for depth
        g2d.setColor(new Color(80, 80, 100, 100));
        for (int i = 1; i <= 5; i++) {
            double t = i / 5.0;
            int y = (int) (150 + (450 - 150) * t);
            int leftX = (int) (200 + (100 - 200) * t);
            int rightX = (int) (400 + (500 - 400) * t);
            g2d.drawLine(leftX, y, rightX, y);
        }

        // Dynamic light source from spell
        if (animationFrame > 60 && animationFrame < 120 && isCasting) {
            RadialGradientPaint light = new RadialGradientPaint(300, 300, 150, new float[]{0f, 1f},
                    new Color[]{new Color(255, 255, 200, 100), new Color(255, 255, 200, 0)});
            g2d.setPaint(light);
            g2d.fill(new Ellipse2D.Double(150, 150, 300, 300));
        }

        // Draw caster wizard (left, with animation)
        double casterScale = (animationFrame > 20 && animationFrame < 60 && isCasting) ? 1.1 : 1.0; // Slight zoom
        drawWizard(g2d, 150, 400, Color.BLUE, true, casterScale);
        // Wand with 3D effect
        if (animationFrame > 20 && animationFrame < 40 && isCasting) {
            RadialGradientPaint wandGradient = new RadialGradientPaint(200, 350, 10, new float[]{0f, 1f},
                    new Color[]{Color.WHITE, Color.YELLOW});
            g2d.setPaint(wandGradient);
            g2d.setStroke(new BasicStroke(4));
            g2d.drawLine(200, 350, 240, 310); // Raised wand
        } else {
            g2d.setColor(new Color(80, 80, 80));
            g2d.setStroke(new BasicStroke(4));
            g2d.drawLine(200, 350, 200, 350);
        }

        // Draw target wizard (right)
        drawWizard(g2d, 450, 400, Color.RED, false, 1.0);

        // Health bars
        drawHealthBar(g2d, 150, 300, casterHealth, Color.GREEN);
        drawHealthBar(g2d, 450, 300, targetHealth, Color.GREEN);

        // Spell spark
        if (animationFrame > 40 && animationFrame < 60 && isCasting) {
            RadialGradientPaint sparkGradient = new RadialGradientPaint(250, 300, 5, new float[]{0f, 1f},
                    new Color[]{Color.WHITE, Color.YELLOW});
            g2d.setPaint(sparkGradient);
            g2d.setStroke(new BasicStroke(6));
            g2d.drawLine(200, 300, 300, 300);
        }

        // Power Nova animation
        if (animationFrame > 60 && animationFrame < 100 && isCasting) {
            int size = (animationFrame - 60) * 15; // Faster expansion
            RadialGradientPaint novaGradient = new RadialGradientPaint(300, 300, size / 2, new float[]{0f, 0.7f, 1f},
                    new Color[]{Color.WHITE, new Color(255, 165, 0), new Color(255, 165, 0, 0)});
            g2d.setPaint(novaGradient);
            g2d.fill(new Ellipse2D.Double(300 - size / 2, 300 - size / 2, size, size));
            // Concentric glow
            g2d.setColor(new Color(255, 165, 0, 80));
            g2d.fill(new Ellipse2D.Double(300 - size / 2 - 30, 300 - size / 2 - 30, size + 60, size + 60));
        } else if (animationFrame >= 100 && animationFrame < 120 && isCasting) {
            // Explosion with particles
            RadialGradientPaint explosionGradient = new RadialGradientPaint(300, 300, 150, new float[]{0f, 1f},
                    new Color[]{Color.RED, new Color(255, 255, 0)});
            g2d.setPaint(explosionGradient);
            g2d.fill(new Ellipse2D.Double(150, 150, 300, 300));
            // Particle effects
            g2d.setColor(Color.YELLOW);
            for (int i = 0; i < 10; i++) {
                double angle = Math.random() * 2 * Math.PI;
                int dist = (int) (Math.random() * 100);
                int px = (int) (300 + Math.cos(angle) * dist);
                int py = (int) (300 + Math.sin(angle) * dist);
                g2d.fillOval(px - 3, py - 3, 6, 6);
            }
            if (animationFrame == 100) {
                targetHealth -= 47;
                if (targetHealth < 0) targetHealth = 0;
            }
        }

        // Defeated message
        if (targetHealth <= 0) {
            g2d.setColor(Color.WHITE);
            g2d.setFont(new Font("Arial", Font.BOLD, 24));
            g2d.drawString("Target Defeated!", 200, 100);
        }
    }

    private void drawWizard(Graphics2D g2d, int x, int y, Color color, boolean isCaster, double scale) {
        // Shadow
        g2d.setColor(new Color(0, 0, 0, 100));
        g2d.fill(new Ellipse2D.Double(x - 15, y + 30, 80 * scale, 30));

        // Wizard robe (conical shape)
        GeneralPath robe = new GeneralPath();
        robe.moveTo(x, y - 50 * scale);
        robe.lineTo(x - 30 * scale, y + 50 * scale);
        robe.lineTo(x + 30 * scale, y + 50 * scale);
        robe.closePath();
        GradientPaint robeGradient = new GradientPaint(x, y - 50, color.brighter(), x, y + 50, color.darker());
        g2d.setPaint(robeGradient);
        g2d.fill(robe);

        // Head
        g2d.fill(new Ellipse2D.Double(x - 25 * scale, y - 80 * scale, 50 * scale, 50 * scale));

        // Magical aura
        if (isCasting && isCaster) {
            g2d.setColor(new Color(color.getRed(), color.getGreen(), color.getBlue(), 50));
            int auraSize = 60 + (int) (Math.sin(animationFrame * 0.1) * 10);
            g2d.fill(new Ellipse2D.Double(x - auraSize / 2, y - auraSize / 2, auraSize, auraSize));
        }
    }

    private void drawHealthBar(Graphics2D g2d, int x, int y, int health, Color color) {
        g2d.setColor(Color.BLACK);
        g2d.fillRect(x - 3, y - 3, 106, 26);
        RadialGradientPaint healthGradient = new RadialGradientPaint(x + 50, y + 10, 50, new float[]{0f, 1f},
                new Color[]{color.brighter(), color.darker()});
        g2d.setPaint(healthGradient);
        g2d.fillRect(x, y, health, 20);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if (isCasting) {
            animationFrame++;
            if (animationFrame >= 120) {
                animationFrame = 0;
                isCasting = false;
            }
        }
        repaint();
    }

    @Override
    public void keyPressed(KeyEvent e) {
        if (e.getKeyCode() == KeyEvent.VK_SPACE && !isCasting && targetHealth > 0) {
            isCasting = true;
            animationFrame = 0;
        }
    }

    @Override
    public void keyReleased(KeyEvent e) {}

    @Override
    public void keyTyped(KeyEvent e) {}

    public static void main(String[] args) {
        JFrame frame = new JFrame("Wizard101 Power Nova 3D Battle Simulation");
        WizardBattleSimulation panel = new WizardBattleSimulation();
        frame.add(panel);
        frame.setSize(600, 600);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
        panel.requestFocusInWindow();
    }
}