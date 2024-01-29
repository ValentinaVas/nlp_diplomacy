# Instalación y ejecución del modelo:


1. Realizar la instalación del ambiente virtual dentro de la carpeta del Orquestador escribiendo en la consola: 
    >> "python -m venv .venv"


2. Activar el ambiente virtual
    >> ".venv\Scripts\activate"

3. Actualizar el Bibliotecario (Importante para mitigar conflictos de librerías):

    >> "python -m pip install --upgrade pip"

    Si se desea verificar antes la versión se puede ejecutar el comando:

    >> "pip list"

    Nota: Es importante que la actualización se haga desde el artifactory, para ello lo más práctico es tener un archivo llamado pip.ini dentro de una carpeta llamada pip en la carpeta del usuario del computador. El archivo pip.ini debe contener lo siguiente:

    [global]
    index-url=https://artifactory.apps.bancolombia.com/api/pypi/pypi-bancolombia/simple
    trusted-host=artifactory.apps.bancolombia.com
    user=false

4. Instalar los paquetes que requiere el orquetador escribiendo en la consola: 
    >> "pip install -e." (Más rápido) o 
    >> "pip install --no-cache-dir -e." (En caso que no funcione la anterior probar con esta)

5. Escribir el DSN y usuario en el archivo config.json, ubicado en la carpeta: 
    src > static > config.json.

6. Para ejecutar el proyecto hay que ubicarse en la carpeta donde está el archivo **script_ejecución.py** y ejecutar en la consola el siguiente comando:
    
    >> python script_ejecucion.py


