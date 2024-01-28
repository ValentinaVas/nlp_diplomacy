# Instalación y ejecución del modelo:


5. Realizar la instalación del ambiente virtual dentro de la carpeta del Orquestador escribiendo en la consola: 
    >> "python -m venv .venv"


6. Activar el ambiente virtual
    >> ".venv\Scripts\activate"

7. Actualizar el Bibliotecario (Importante para mitigar conflictos de librerías):

    >> "python -m pip install --upgrade pip"

    Si se desea verificar antes la versión se puede ejecutar el comando:

    >> "pip list"

    Nota: Es importante que la actualización se haga desde el artifactory, para ello lo más práctico es tener un archivo llamado pip.ini dentro de una carpeta llamada pip en la carpeta del usuario del computador. El archivo pip.ini debe contener lo siguiente:

    [global]
    index-url=https://artifactory.apps.bancolombia.com/api/pypi/pypi-bancolombia/simple
    trusted-host=artifactory.apps.bancolombia.com
    user=false

8. Instalar los paquetes que requiere el orquetador escribiendo en la consola: 
    >> "pip install -e." (Más rápido) o 
    >> "pip install --no-cache-dir -e." (En caso que no funcione la anterior probar con esta)

9. Escribir el DSN y usuario en el archivo config.json, ubicado en la carpeta: 
    src > static > config.json.

10. Para ejecutar el orquestador hay que ubicarse en la carpeta donde está el archivo **script_ejecución.py** y ejecutar en la consola el siguiente comando:
    
    >> python script_ejecucion.py

Y listo a disfrutar del Orquestador!!!

PLUS ->> 
1. (Ingreso automático de la contraseña): Si se va a ejecutar muchas vecees en el día el orquestador, es posible que se desee que la contraseña sea ingresada de manera automática esta puede colocarse en las variables de entorno de la cuenta y ser llamada mediante el código <<os.getenv("pass")>>, donde **pass** es el nombre asignado a la variable de entorno de la cuenta al que su valor hay que escribir la contraseña, el código se debe colocar donde es solicitada la contraseña en la librería del orquestador (orquestador2 > orquestador.py > "password" > None > os.getenv("pass")). Para que funcione la primera vez que se instancia la variable de entorno hay que reiniciar todo el Visual Studio Code. <Claramente este paso implica un punto de seguiridad a tener en cuenta, y debería ser empleado con toda la cautela posible y eliminar la contraseña de las variables de entorno si esta no se va a emplear para evitar cualquier riesgo de seguridad de la información.>

2. Ejecución automática del orquestador a partir de un archivo Bat: Es posible ejecutar el orquestador a partir de un archivo .bat, en el cual hay que escribir lo siguiente:

    >> call .venv/Scripts/activate
    >> python script_ejecucion.py

Con lo anterior se ejecutará el orquestador de manera automática y correrá su programa, de manera opcionals si se desea ver la consola de comandos se puede colocar el comando <<pause>> al final.
