#!/usr/bin/env python3
import sys
import xml.etree.ElementTree as ET

def indent(elem, level=0):
    """Pretty print XML tree for Python <3.9"""
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def convert_ros_to_drake(ros_urdf_path, drake_urdf_path, mesh_base="../meshes/"):
    tree = ET.parse(ros_urdf_path)
    root = tree.getroot()

    # Remove xacro namespace if present
    for elem in root.findall(".//*"):
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    # Remove unsupported tags
    for tag in ["safety_controller", "dynamics"]:
        for elem in root.findall(f".//{tag}"):
            for parent in root.iter():
                if elem in list(parent):
                    parent.remove(elem)

    # Fix mesh paths and add Drake proximity properties
    for geom in root.findall(".//geometry/mesh"):
        filename = geom.attrib.get("filename", "")
        if filename.startswith("package://"):
            geom.attrib["filename"] = mesh_base + filename.split("/")[-1]

        # Find collision parent
        collision = None
        for parent in root.iter():
            if geom in list(parent):
                for grand in root.iter():
                    if parent in list(grand):
                        if grand.tag == "collision":
                            collision = grand
                            break
        if collision is not None:
            props = ET.SubElement(collision, "drake:proximity_properties")
            ET.SubElement(props, "drake:rigid_hydroelastic")
            ET.SubElement(props, "drake:mesh_resolution_hint", value="1.5")
            ET.SubElement(props, "drake:hunt_crossley_dissipation", value="1.25")
            ET.SubElement(props, "drake:mu_dynamic", value="0.8")
            ET.SubElement(props, "drake:mu_static", value="0.8")

    # Add transmission extras if transmissions exist
    for trans in root.findall(".//transmission"):
        actuator = trans.find("actuator")
        if actuator is not None:
            ET.SubElement(actuator, "drake:gear_ratio", value="100.0")
            ET.SubElement(actuator, "drake:rotor_inertia", value="0.00005")

    # Add collision filter group (disable self-collision)
    filter_group = ET.Element("drake:collision_filter_group", name="all")
    for link in root.findall("link"):
        ET.SubElement(filter_group, "drake:member", link=link.attrib["name"])
    ET.SubElement(filter_group, "drake:ignored_collision_filter_group", name="all")
    root.append(filter_group)

    # Pretty print for older Python
    indent(root)

    # Save new URDF
    tree.write(drake_urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"Converted URDF saved to {drake_urdf_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 convert_urdf.py input.urdf output.urdf")
        sys.exit(1)

    convert_ros_to_drake(sys.argv[1], sys.argv[2])

