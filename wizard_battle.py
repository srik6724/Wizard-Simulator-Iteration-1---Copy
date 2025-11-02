from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    Geom,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    GeomTriangles,
    GeomNode,
    NodePath,
    PointLight,
    Point3,
    Vec3,
    GeomSphere
)
import math
import random

class WizardBattle3DSimulation(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()  # Disable default camera controls
        self.setBackgroundColor(0, 0, 0.2, 1)  # Dark blue background

        # Camera setup
        self.camera.setPos(0, -1000, 200)
        self.camera.setHpr(0, -15, 0)  # Slight downward angle

        # Animation variables
        self.animation_frame = 0
        self.caster_health = 100
        self.target_health = 100
        self.is_casting = False
        self.particles = []

        # Arena (tilted plane)
        plane_format = GeomVertexFormat.getV3n3c4()
        plane_vdata = GeomVertexData("plane", plane_format, Geom.UHStatic)
        plane_writer = GeomVertexWriter(plane_vdata, "vertex")
        plane_normal = GeomVertexWriter(plane_vdata, "normal")
        plane_color = GeomVertexWriter(plane_vdata, "color")
        plane_writer.addData3(-200, 0, -150)
        plane_writer.addData3(200, 0, -150)
        plane_writer.addData3(150, 0, -450)
        plane_writer.addData3(-150, 0, -450)
        for _ in range(4):
            plane_normal.addData3(0, 1, 0)
            plane_color.addData4(0.5, 0.5, 0.5, 1)  # Gray
        plane_tris = GeomTriangles(Geom.UHStatic)
        plane_tris.addVertices(0, 1, 2)
        plane_tris.addVertices(0, 2, 3)
        plane_geom = Geom(plane_vdata)
        plane_geom.addPrimitive(plane_tris)
        plane_node = GeomNode("plane")
        plane_node.addGeom(plane_geom)
        self.arena = self.render.attachNewNode(plane_node)
        self.arena.setTwoSided(True)

        # Caster wizard (left)
        self.caster = self.create_wizard(0, 0, 1)  # Blue
        self.caster.setPos(-150, 0, -100)
        self.caster.reparentTo(self.render)

        # Target wizard (right)
        self.target = self.create_wizard(1, 0, 0)  # Red
        self.target.setPos(150, 0, -100)
        self.target.reparentTo(self.render)

        # Health bars
        self.caster_health_bar = self.create_health_bar(self.caster_health)
        self.caster_health_bar.setPos(-150, 0, 50)
        self.caster_health_bar.reparentTo(self.render)

        self.target_health_bar = self.create_health_bar(self.target_health)
        self.target_health_bar.setPos(150, 0, 50)
        self.target_health_bar.reparentTo(self.render)

        # Nova sphere
        nova_node = GeomNode("nova_sphere")
        nova_node.addGeom(GeomSphere(0, 0, 0, 1, 20, 20))
        self.nova_sphere = self.render.attachNewNode(nova_node)
        self.nova_sphere.setScale(0)
        self.nova_sphere.setColor(1, 0.5, 0, 1)  # Orange
        self.nova_sphere.setPos(0, 0, -100)

        # Spell light
        self.spell_light = PointLight("spell_light")
        self.spell_light.setColor((1, 1, 1, 1))
        self.spell_light_node = self.render.attachNewNode(self.spell_light)
        self.spell_light_node.setPos(0, 0, -100)
        self.render.setLight(self.spell_light_node)
        self.spell_light_node.hide()

        # Defeated text
        self.defeated_text = None

        # Key input
        self.accept("space", self.start_cast)

        # Animation task
        self.taskMgr.add(self.update_animation, "update_animation")

    def create_cylinder_geom(self, radius, height, segments=20):
        format = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData("cylinder", format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        color = GeomVertexWriter(vdata, "color")

        # Top and bottom circles
        for i in range(segments):
            angle = i * 2 * math.pi / segments
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            # Bottom vertices
            vertex.addData3(x, y, 0)
            normal.addData3(x / radius, y / radius, 0)
            # Top vertices
            vertex.addData3(x, y, height)
            normal.addData3(x / radius, y / radius, 0)

        # Side vertices (duplicate for distinct normals)
        for i in range(segments):
            angle = i * 2 * math.pi / segments
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            vertex.addData3(x, y, 0)
            normal.addData3(x / radius, y / radius, 0)
            vertex.addData3(x, y, height)
            normal.addData3(x / radius, y / radius, 0)

        # Triangles
        tris = GeomTriangles(Geom.UHStatic)
        # Bottom cap
        for i in range(segments):
            next_i = (i + 1) % segments
            tris.addVertices(i * 2, next_i * 2, segments * 2)  # Center bottom (add later)
        # Top cap
        for i in range(segments):
            next_i = (i + 1) % segments
            tris.addVertices(i * 2 + 1, segments * 2 + 1, next_i * 2 + 1)  # Center top
        # Sides
        for i in range(segments):
            next_i = (i + 1) % segments
            base = segments * 2 + i * 2
            next_base = segments * 2 + (next_i % segments) * 2
            tris.addVertices(base, base + 1, next_base + 1)
            tris.addVertices(base, next_base + 1, next_base)

        # Center vertices for caps
        vertex.addData3(0, 0, 0)  # Bottom center
        normal.addData3(0, 0, -1)
        vertex.addData3(0, 0, height)  # Top center
        normal.addData3(0, 0, 1)

        return vdata, tris

    def create_wizard(self, r, g, b):
        wizard = NodePath("wizard")
        # Robe (cylinder)
        robe_vdata, robe_tris = self.create_cylinder_geom(20, 50)
        robe_color = GeomVertexWriter(robe_vdata, "color")
        for _ in range(robe_vdata.getNumVertices()):
            robe_color.addData4(r, g, b, 1)
        robe_geom = Geom(robe_vdata)
        robe_geom.addPrimitive(robe_tris)
        robe_node = GeomNode("robe")
        robe_node.addGeom(robe_geom)
        robe = wizard.attachNewNode(robe_node)
        # Head (sphere)
        head = wizard.attachNewNode(GeomNode("head"))
        head_node = GeomNode("head")
        head_node.addGeom(GeomSphere(0, 0, 60, 15, 20, 20))
        head.replaceNode(head_node)
        head.setColor(r * 1.2, g * 1.2, b * 1.2, 1)
        # Wand (smaller cylinder)
        wand_vdata, wand_tris = self.create_cylinder_geom(2, 15)
        wand_color = GeomVertexWriter(wand_vdata, "color")
        for _ in range(wand_vdata.getNumVertices()):
            wand_color.addData4(0.6, 0.4, 0.2, 1)  # Brown
        wand_geom = Geom(wand_vdata)
        wand_geom.addPrimitive(wand_tris)
        wand_node = GeomNode("wand")
        wand_node.addGeom(wand_geom)
        wand = wizard.attachNewNode(wand_node)
        wand.setPos(20, 0, 30)
        wand.setHpr(0, 0, 45)
        # Aura (semi-transparent sphere)
        aura = wizard.attachNewNode(GeomNode("aura"))
        aura_node = GeomNode("aura")
        aura_node.addGeom(GeomSphere(0, 0, 0, 40, 20, 20))
        aura.replaceNode(aura_node)
        aura.setColor(r, g, b, 0.2)
        aura.setTwoSided(True)
        return wizard

    def create_health_bar(self, health):
        vdata, tris = self.create_cylinder_geom(2, health)
        color = GeomVertexWriter(vdata, "color")
        for _ in range(vdata.getNumVertices()):
            color.addData4(0, 1, 0, 1)  # Green
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("health_bar")
        node.addGeom(geom)
        bar = self.render.attachNewNode(node)
        bar.setHpr(90, 0, 0)
        return bar

    def start_cast(self):
        if not self.is_casting and self.target_health > 0:
            self.is_casting = True
            self.animation_frame = 0
            self.nova_sphere.setScale(0)
            self.spell_light_node.hide()
            for particle in self.particles:
                particle.removeNode()
            self.particles.clear()

    def update_animation(self, task):
        if not self.is_casting:
            return Task.cont

        self.animation_frame += 1
        wand = self.caster.find("**/wand")
        aura = self.caster.find("**/aura")

        # Wand animation
        if 20 < self.animation_frame < 40:
            wand.setHpr(0, 0, 90)  # Raise wand
            aura.setScale(40 + math.sin(self.animation_frame * 0.2) * 5)
        else:
            wand.setHpr(0, 0, 45)
            aura.setScale(40)

        # Spark
        if self.animation_frame == 40:
            spark = self.render.attachNewNode(GeomNode("spark"))
            spark_node = GeomNode("spark")
            spark_node.addGeom(GeomSphere(-150, 0, -100, 5, 20, 20))
            spark.replaceNode(spark_node)
            spark.setColor(1, 1, 0, 1)  # Yellow
            self.particles.append(spark)
            spark.setPosHprScaleInterval(1.0, Point3(0, 0, -100), Vec3(0, 0, 0), spark.getScale()).start()
            spark.setPosHprScaleInterval(1.0, Point3(0, 0, -100), Vec3(0, 0, 0), Vec3(0, 0, 0)).start()

        # Power Nova sphere
        if 60 < self.animation_frame < 100:
            radius = (self.animation_frame - 60) * 3
            self.nova_sphere.setScale(radius)
            self.nova_sphere.setZ(-100 - radius * 0.5)
            self.spell_light_node.setPos(0, 0, -100 - radius)
            self.spell_light_node.show()
        elif 100 <= self.animation_frame < 120:
            self.nova_sphere.setScale(0)
            self.spell_light_node.hide()
            if self.animation_frame == 100:
                self.target_health -= 47
                if self.target_health < 0:
                    self.target_health = 0
                self.target_health_bar.removeNode()
                self.target_health_bar = self.create_health_bar(self.target_health)
                self.target_health_bar.setPos(150, 0, 50)
                self.target_health_bar.reparentTo(self.render)
                # Explosion particles
                for _ in range(20):
                    particle = self.render.attachNewNode(GeomNode("particle"))
                    particle_node = GeomNode("particle")
                    particle_node.addGeom(GeomSphere(0, 0, -100, 3, 20, 20))
                    particle.replaceNode(particle_node)
                    particle.setColor(1, 1, 0, 1)
                    self.particles.append(particle)
                    angle = random.random() * 2 * math.pi
                    dist = random.random() * 200
                    particle.setPosHprScaleInterval(
                        1.0,
                        Point3(dist * math.cos(angle), 0, -100 + dist * math.sin(angle)),
                        Vec3(0, 0, 0),
                        particle.getScale()
                    ).start()
                    particle.setPosHprScaleInterval(
                        1.0,
                        Point3(dist * math.cos(angle), 0, -100 + dist * math.sin(angle)),
                        Vec3(0, 0, 0),
                        Vec3(0, 0, 0)
                    ).start()

        elif self.animation_frame >= 120:
            self.is_casting = False
            self.animation_frame = 0

        # Defeated message
        if self.target_health <= 0 and not self.defeated_text:
            self.defeated_text = OnscreenText(
                text="Target Defeated!",
                pos=(0, 0.3),
                fg=(1, 1, 1, 1),
                scale=0.1
            )

        return Task.cont

if __name__ == "__main__":
    app = WizardBattle3DSimulation()
    app.run()