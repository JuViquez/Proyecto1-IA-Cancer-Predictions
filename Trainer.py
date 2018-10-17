import argparse
from source.Program import Program

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    parser.add_argument("--prefijo", required = True, help = 'nombre de archivo del dataset')
    parser.add_argument("--indice_columna_y", required = True, type = int, help = 'indice columna de observacion,  el minimo es 0')
    parser.add_argument("--porcentaje-pruebas", required = True, type = float, help = 'porcentaje del dataset que se utilizara para pruebas')
    
    arbol_parser = subparsers.add_parser('arbol', help='modelo random forest')
    arbol_parser.add_argument("--umbral-poda", required = True, type = float, nargs = "?", default = 0, help = 'ganancia de informacion necesaria para realizar una poda en el arbol' )
    arbol_parser.set_defaults(arbol = True)
    arbol_parser.set_defaults(red_neuronal = False)

    red_parser = subparsers.add_parser('red-neuronal', help='modelo red neuronal')
    red_parser.add_argument("--numero-capas", type = int, help = 'numero de capas de la red neuronal' )
    red_parser.add_argument("--unidades-por-capa", type = int,  help = 'neuronas por capa' )
    red_parser.add_argument("--funcion-activacion", type = int,  help = 'funcion de activacion de las neuronas' )
    red_parser.add_argument("--funcion-activacion-salida", type = int, help = 'funcion de activacion de las neuronas de salida' )
    red_parser.set_defaults(red_neuronal = True)
    red_parser.set_defaults(arbol = False)
    args = parser.parse_args()
    
    program = Program()
    program.main(args)
    

if __name__ == "__main__":
    main()