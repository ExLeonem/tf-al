

class FileHandler:
    """
        Base class for different file handlers.
        Offers generic handler utilities.
    """

    def add(self, filename, data):
        """
            Adding new data to already existing one.

            Parameters:
                filename (str): The filename to append data to.
                data (dict): new data to append to the old one.
        """
        pass

    
    def read(self, filename):
        """
            Reading written data.

            Parameters:
                filename (str): The file to read data from.
            
            Returns:
                (dict) The data.
        """
        pass

    
    # ----
    # Utilities
    # -----------------

    def _add_extension(self, filename, ext):
        """
            Adds an extension to a filename.

            Parameters:
                filename (str): The filename to check for the extension
                ext (str): The file extension to add and check for
            
            Returns:
                (str) the file name with a file extension appended. 
        """

        if ext not in filename:
            return filename + "." + ext
        
        return filename