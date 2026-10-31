# VM Image Store

Versioned VM image catalog for trajectory collection (paper Sec. 2.1.3).
Every rollout episode is instantiated from a catalog entry — typically an
immutable base image plus a disposable writable overlay. Large disks can be
hardlinked or symlinked to avoid duplication; `catalog.json` keeps the stable
image id, OS descriptor, and software inventory.

Directory layout:

```text
vm_images/
  catalog.json
  linux/<image-id>/{disk.qcow2,descriptor.json}
  windows/<image-id>/{disk.qcow2,descriptor.json}
  macos/<image-id>/{disk.qcow2,descriptor.json}
```

The shipped `catalog.json` contains placeholder example entries only; disk
images are not distributed with this repository. Register your own images:

```bash
cuamaster-vm-images --store vm_images list
cuamaster-vm-images --store vm_images verify
cuamaster-vm-images --store vm_images register \
  --id my-ubuntu-v2 --os-type Ubuntu \
  --os-version 24.04 --source /path/to/image.qcow2 --mode link
```

Registration modes: `hardlink` and `link` use no extra disk; `copy` duplicates
the image; `reference` only records the source path in the catalog.

Every catalog entry must point to a `descriptor.json`. A descriptor produced by
a real boot inventory records the guest OS type, version, and architecture plus
the normalized visible applications with their versions, package managers,
package names, executables, and categories. Images that only passed artifact
verification must set `"inventory_status": "pending"` explicitly — an empty app
list must not be presented as a real inventory.

Ubuntu/Windows images can be booted locally through the public OSWorld Docker
launcher:

```bash
vm_images/manage_osworld_docker.sh start <image-id>
```

macOS images boot through the KVM + OpenCore path
(`cuamaster.collection.macos_kvm_launcher`) with the OpenCore/controller
overlay assets placed under `vm_images/macos/runtime/` (not distributed).

Base images can be derived programmatically — e.g. removing randomly selected
applications or reorganizing desktop layouts — to expand environment coverage
at low marginal cost; register each variant as its own catalog entry.
