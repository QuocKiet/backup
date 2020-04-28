import wx
import nhandang
import cv2

class Example(wx.Frame):
	def __init__(self, parent, title):
		super(Example, self).__init__(parent, title=title, size=(960, 500))
		self.Centre()
		self.panel = wx.Panel(self)
		self.listImage = []
		vbox = wx.BoxSizer(wx.VERTICAL) 
		hbox1 = wx.BoxSizer(wx.HORIZONTAL)


		self.btnFile = wx.Button(self.panel, -1, "Choose File")
		self.btnCreate = wx.Button(self.panel, -1, "Get License Plate")
		self.textChoose = wx.StaticText(self.panel, -1, "File choosen: ")
		self.textLicense = wx.StaticText(self.panel, -1, "License Plate Image: ")
		self.textLicense2 = wx.StaticText(self.panel, -1, "License Plate Text: ")
		self.textCtrlLicense = wx.TextCtrl(self.panel, -1, value="", size=(150,40))

		image = wx.Image('logo.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
		imageBitmap = wx.StaticBitmap(self.panel, wx.ID_ANY, wx.Bitmap(image))
		imageBitmap.SetPosition((25,25))


		self.btnFile.Bind(wx.EVT_BUTTON, self.openFile)
		self.btnCreate.Bind(wx.EVT_BUTTON, self.recognize)

		self.btnFile.SetPosition((115, 200))
		self.btnCreate.SetPosition((215, 200))
		self.btnCreate.Disable()
		self.textChoose.SetPosition((640, 25))
		self.textChoose.Hide()
		self.textLicense.SetPosition((70, 250))
		self.textLicense.Hide()
		self.textLicense2.SetPosition((70, 370))
		self.textLicense2.Hide()
		self.textCtrlLicense.SetPosition((70, 390))
		self.textCtrlLicense.Hide()
		myFont = wx.Font(20, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False)
		self.textCtrlLicense.SetFont(myFont)


		hbox1.Add(self.btnFile)
		vbox.Add(hbox1)

	def openFile(self, event):
		openFileDialog = wx.FileDialog(self.panel, "Open", wildcard = "Image Files Only (*.jpg,*.png,*.jpeg)|*.jpg;*.png;*.jpeg")
		openFileDialog.ShowModal()
		path = openFileDialog.GetPath()
		self.listImage.append(path)

		# self.choose = wx.ListBox(self.panel, wx.ID_ANY, wx.Point(120,300), wx.Size(300, 300), self.listImage, style = wx.LB_HSCROLL)
		self.textChoose.Show()
		openFileDialog.Destroy()
		image = wx.Image(path, wx.BITMAP_TYPE_ANY)
		image = image.Scale(480, 360, wx.IMAGE_QUALITY_HIGH)
		imageBitmap = wx.StaticBitmap(self.panel, wx.ID_ANY, wx.Bitmap(image))
		imageBitmap.SetPosition((440,70))
		self.btnCreate.Enable()

	def recognize(self, event):
		plate = ""
		try:
			plate = nhandang.lpr(self.listImage[0])
		except Exception as e:
			pass
		# print(type(image))

		if len(plate) == 0:
			image = wx.Image('notfound.png', wx.BITMAP_TYPE_ANY)
			image = image.Scale(300, 60, wx.IMAGE_QUALITY_HIGH)
			imageBitmap = wx.StaticBitmap(self.panel, wx.ID_ANY, wx.Bitmap(image))
			imageBitmap.SetPosition((70, 270))
			self.textCtrlLicense.SetValue("Not Found")

		else:		
			image = wx.Image('Result.jpg', wx.BITMAP_TYPE_ANY)
			image = image.Scale(300, 60, wx.IMAGE_QUALITY_HIGH)
			imageBitmap = wx.StaticBitmap(self.panel, wx.ID_ANY, wx.Bitmap(image))
			imageBitmap.SetPosition((70, 270))
			self.textCtrlLicense.SetValue(plate)

		self.textLicense.Show()
		self.textLicense2.Show()
		self.textCtrlLicense.Show()
		self.listImage = []

def main():

    app = wx.App()
    ex = Example(None, title = "License Plate Recognition")
    ex.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()